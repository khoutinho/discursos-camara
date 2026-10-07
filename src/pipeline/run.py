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
"""

import argparse
import sys
from pathlib import Path

import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from .preprocessing import load_and_clean, classify_ideology, classify_region
from .topic_modeling import TopicModeler, TopicModelConfig
from .statistics import compute_all_statistics
from .export import export_dashboard_json, export_processed_csv


def run_pipeline(
    input_csv: str,
    output_dir: str = 'output',
    nr_topics: int | None = None,
    threshold: float = 0.1,
    cache_dir: str = '.cache',
    encoding: str = 'utf-8-sig',
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Carregamento e classificação ──
    print("=" * 60)
    print("1. CARREGAMENTO E CLASSIFICAÇÃO")
    print("=" * 60)

    df = load_and_clean(input_csv, encoding=encoding)
    print(f"Registros carregados: {len(df)}")

    df = classify_ideology(df)
    df = classify_region(df)

    print(f"\nEspectro (3 classes):")
    for esp, cnt in df['Espectro'].value_counts().items():
        print(f"  {esp}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── 2. Topic Modeling ──
    print("\n" + "=" * 60)
    print("2. TOPIC MODELING (BERTopic + HDBSCAN)")
    print("=" * 60)

    df_with_text = df[df['Discurso'].notna()].copy()
    print(f"Discursos com texto: {len(df_with_text)}")

    config = TopicModelConfig(
        nr_topics=nr_topics,
        multi_label_threshold=threshold,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    modeler = TopicModeler(config)
    results = modeler.fit(df_with_text['Discurso'])

    topics = results['topics']
    topic_labels = results['topic_labels']
    multi_labels = results['multi_labels']

    df_with_text['Topico'] = topics
    df_with_text['Topico_label'] = df_with_text['Topico'].map(topic_labels)
    df_with_text['Topicos_multi'] = ['; '.join(labels) for labels in multi_labels]

    n_topics = len([t for t in topic_labels if t != -1])
    n_noise = int((topics == -1).sum())
    print(f"\nTópicos encontrados: {n_topics}")
    print(f"Documentos em ruído: {n_noise} ({n_noise/len(topics)*100:.1f}%)")
    print(f"\nTop tópicos:")
    for label, count in (
        df_with_text[df_with_text['Topico'] != -1]['Topico_label']
        .value_counts().head(10).items()
    ):
        print(f"  {label}: {count} discursos")

    # ── 3. Análises estatísticas ──
    print("\n" + "=" * 60)
    print("3. ANÁLISES ESTATÍSTICAS")
    print("=" * 60)

    stats = compute_all_statistics(
        df_with_text,
        topics=topics,
        topic_labels=topic_labels,
    )

    mk = stats['mann_kendall_geral']
    print(f"Mann-Kendall geral: {mk['tendencia']} (p={mk['p']}, τ={mk['tau']})")

    for item in stats['mann_kendall_por_espectro']:
        print(f"  {item['espectro']}: {item['tendencia']} (p={item['p']}, τ={item['tau']})")

    # ── 4. Exportação ──
    print("\n" + "=" * 60)
    print("4. EXPORTAÇÃO")
    print("=" * 60)

    export_dashboard_json(stats, output_path / 'dashboard_data.json')
    export_processed_csv(df_with_text, output_path / 'discursos_processados.csv')

    print("\n✅ Pipeline concluído!")
    print(f"Resultados em: {output_path.absolute()}")

    return df_with_text, stats, modeler


def main():
    parser = argparse.ArgumentParser(description='Pipeline NLP — Discursos da Câmara')
    parser.add_argument('--input', required=True, help='CSV de entrada')
    parser.add_argument('--output', default='output', help='Diretório de saída')
    parser.add_argument('--nr-topics', type=int, default=None, help='Número alvo de tópicos')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold multi-label')
    parser.add_argument('--cache-dir', default='.cache', help='Cache de embeddings')
    parser.add_argument('--encoding', default='utf-8-sig', help='Encoding do CSV')
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        output_dir=args.output,
        nr_topics=args.nr_topics,
        threshold=args.threshold,
        cache_dir=args.cache_dir,
        encoding=args.encoding,
    )


if __name__ == '__main__':
    main()
