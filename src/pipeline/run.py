#!/usr/bin/env python3
"""
Orquestrador do pipeline NLP.

Uso:
    python -m src.pipeline.run --input csvs/discursos_completo2.csv --output output/

Opções:
    --input         Caminho do CSV de entrada
    --output        Diretório de saída (default: output/)
    --nr-topics     Número alvo de tópicos pro reduce_topics (default: automático)
    --threshold     Threshold de probabilidade pra multi-label (default: 0.1)
    --cache-dir     Diretório pra cache de embeddings (default: .cache/)
    --encoding      Encoding do CSV (default: utf-8-sig)
    --keywords      Keywords da busca, separadas por vírgula (tabela keyword × tópico)
"""

import argparse
from datetime import date
from pathlib import Path

import nltk
import numpy as np
import pandas as pd

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from .preprocessing import load_and_clean, classify_ideology, classify_region
from .model_config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from .text_cleaning import MODEL_TEXT_COLUMN, add_model_text
from .topic_modeling import TopicModeler, TopicModelConfig
from .statistics import compute_all_statistics
from .export import export_dashboard_json, export_processed_csv


def analyze(
    df: pd.DataFrame,
    config: TopicModelConfig,
    embeddings: np.ndarray | None = None,
    progress_callback=None,
    period: tuple[date, date] | None = None,
    plenary_totals: dict[str, int | None] | None = None,
    keywords: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Classificação ideológica/regional → tópicos → estatísticas.

    Args:
        df: discursos já passados por preprocessing.clean(), todos com texto.
        config: parâmetros do topic modeling.
        embeddings: embeddings na mesma ordem de df (None = calcula).
        progress_callback: Callable(stage: str, pct: float).
        period: período coletado (meses sem discursos contam como zero nas tendências).
        plenary_totals: {'YYYY-MM': total do Plenário}; None = tendências só em contagem.
        keywords: keywords da busca (tabela keyword × tópico, com 2+ keywords).

    Returns:
        (df com colunas de tópico, dict de estatísticas para o dashboard)
    """
    _cb = progress_callback or (lambda *a: None)
    df = add_model_text(classify_region(classify_ideology(df)))

    modeler = TopicModeler(config)
    results = modeler.fit(df[MODEL_TEXT_COLUMN], progress_callback=_cb, embeddings=embeddings)

    topics = results['topics']
    topic_labels = results['topic_labels']
    df = df.copy()
    df['Topico'] = topics
    df['Topico_label'] = df['Topico'].map(topic_labels)
    df['Topico_reatribuido'] = results['reassigned']
    df['Topicos_multi'] = ['; '.join(labels) for labels in results['multi_labels']]

    _cb('statistics', 0.0)
    stats = compute_all_statistics(
        df, topics=topics, topic_labels=topic_labels, period=period, plenary_totals=plenary_totals,
        keywords=keywords,
    )

    n = len(df)
    pct = lambda k: round(k / n * 100, 1) if n else 0.0
    n_noise = int((topics == -1).sum())             # ruído que sobrou (0 com reduce_outliers)
    n_reassigned = int(results['reassigned'].sum())
    n_multi = sum(len(m) > 1 for m in results['multi_labels'])
    stats['modelo'] = {
        'n_discursos': n,
        'n_topicos': len([t for t in topic_labels if t != -1]),
        'n_ruido': n_noise,
        'pct_ruido': pct(n_noise),
        'n_ruido_original': results['n_noise'],     # o que o HDBSCAN deixou sem tópico
        'pct_ruido_original': pct(results['n_noise']),
        'n_reatribuidos': n_reassigned,
        'pct_reatribuidos': pct(n_reassigned),
        'n_multi_label': n_multi,
        'pct_multi_label': pct(n_multi),
    }
    stats['metodologia'] = modeler.methodology()
    return df, stats


def run_pipeline(
    input_csv: str,
    output_dir: str = 'output',
    nr_topics: int | None = None,
    threshold: float = 0.1,
    cache_dir: str = '.cache',
    encoding: str = 'utf-8-sig',
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    keywords: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(input_csv, encoding=encoding)
    df = df[df['Discurso'].notna()].copy()
    print(f"Discursos com texto e data válida: {len(df)}")

    config = TopicModelConfig(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        multi_label_threshold=threshold,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    df, stats = analyze(df, config, progress_callback=lambda stage, _pct: print(f"→ {stage}"), keywords=keywords)

    m = stats['modelo']
    print(f"\nTópicos encontrados: {m['n_topicos']}")
    print(f"Ruído do HDBSCAN: {m['n_ruido_original']} ({m['pct_ruido_original']}%), "
          f"reatribuídos ao tópico mais próximo: {m['n_reatribuidos']}")
    print(f"Discursos com mais de um tópico (multi-label): {m['n_multi_label']} ({m['pct_multi_label']}%)")
    print("\nTop tópicos:")
    for label, count in df[df['Topico'] != -1]['Topico_label'].value_counts().head(10).items():
        print(f"  {label}: {count} discursos")

    print("\nTendências (contagem; a linha de comando não busca os totais do Plenário para a proporção):")
    for serie in stats['tendencias']['series']:
        if serie['grupo'] == 'topico':
            continue
        for teste in ('anual', 'sazonal'):
            r = serie['contagem'][teste]
            texto = f"{r['tendencia']} (p={r['p']}, τ={r['tau']})" if r else 'dados insuficientes'
            print(f"  {serie['nome']} — {teste}: {texto}")

    export_dashboard_json(stats, output_path / 'dashboard_data.json')
    export_processed_csv(df, output_path / 'discursos_processados.csv')
    print(f"\nResultados em: {output_path.absolute()}")
    return df, stats


def main():
    parser = argparse.ArgumentParser(description='Pipeline NLP — Discursos da Câmara')
    parser.add_argument('--input', required=True, help='CSV de entrada')
    parser.add_argument('--output', default='output', help='Diretório de saída')
    parser.add_argument('--nr-topics', type=int, default=None, help='Número alvo de tópicos')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold multi-label')
    parser.add_argument('--cache-dir', default='.cache', help='Cache de embeddings')
    parser.add_argument('--encoding', default='utf-8-sig', help='Encoding do CSV')
    parser.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, choices=list(EMBEDDING_MODELS),
                        help='Modelo de embedding')
    parser.add_argument('--keywords', default=None,
                        help='Keywords da busca, separadas por vírgula (tabela keyword × tópico)')
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        output_dir=args.output,
        nr_topics=args.nr_topics,
        threshold=args.threshold,
        cache_dir=args.cache_dir,
        encoding=args.encoding,
        embedding_model=args.embedding_model,
        keywords=args.keywords.split(',') if args.keywords else None,
    )


if __name__ == '__main__':
    main()
