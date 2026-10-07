"""
Análises estatísticas: Mann-Kendall, Pearson, cosseno, CV, e análises por tópico.
"""

import numpy as np
import pandas as pd
import pymannkendall as mk
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from .config import STOPWORDS_DOMAIN


def compute_all_statistics(
    df: pd.DataFrame,
    topics: np.ndarray | None = None,
    topic_labels: dict[int, str] | None = None,
) -> dict:
    """
    Calcula todas as análises estatísticas.

    Args:
        df: DataFrame com colunas Ano, Espectro, Partido, Regiao, Discurso.
        topics: Array de tópico dominante por documento (mesmo índice que df).
        topic_labels: Mapeamento topic_id → label.

    Returns:
        dict com todas as análises prontas pra exportação.
    """
    stats = {}

    year_min = int(df['Ano'].min())
    year_max = int(df['Ano'].max())
    year_range = range(year_min, year_max + 1)

    # ── Evolução temporal ──
    evo_ano = df.groupby('Ano').size().reset_index(name='Total')
    stats['evolucao_temporal'] = evo_ano.to_dict('records')

    # ── Mann-Kendall geral ──
    mk_geral = mk.original_test(evo_ano['Total'].values)
    stats['mann_kendall_geral'] = {
        'tendencia': mk_geral.trend,
        'p': round(mk_geral.p, 4),
        'tau': round(mk_geral.Tau, 4),
    }

    # ── Mann-Kendall por espectro ──
    mk_por_espectro = []
    espectros = ['Esquerda', 'Centro', 'Direita']
    for esp in espectros:
        serie = (
            df[df['Espectro'] == esp]
            .groupby('Ano').size()
            .reindex(year_range, fill_value=0)
        )
        if len(serie) > 3:
            r = mk.original_test(serie.values)
            mk_por_espectro.append({
                'espectro': esp, 'tendencia': r.trend,
                'p': round(r.p, 4), 'tau': round(r.Tau, 4),
            })
    stats['mann_kendall_por_espectro'] = mk_por_espectro

    # ── Proporção por espectro ao longo do tempo ──
    evo_esp = df.groupby(['Ano', 'Espectro']).size().unstack(fill_value=0)
    evo_esp_pct = evo_esp.div(evo_esp.sum(axis=1), axis=0) * 100

    stats['espectro_por_ano'] = (
        df.groupby(['Ano', 'Espectro']).size()
        .reset_index(name='count')
        .to_dict('records')
    )
    stats['espectro_pct_por_ano'] = (
        evo_esp_pct.reset_index()
        .melt(id_vars='Ano', var_name='espectro', value_name='pct')
        .dropna()
        .to_dict('records')
    )

    # ── Correlação de Pearson entre espectros ──
    pearson_results = []
    cols = [c for c in espectros if c in evo_esp.columns]
    for i, e1 in enumerate(cols):
        for e2 in cols[i+1:]:
            r, p = pearsonr(evo_esp[e1], evo_esp[e2])
            pearson_results.append({
                'par': f'{e1} × {e2}', 'r': round(r, 3), 'p': round(p, 4),
            })
    stats['pearson'] = pearson_results

    # ── Similaridade do cosseno entre espectros ──
    cosseno_results = []
    for i, e1 in enumerate(cols):
        for e2 in cols[i+1:]:
            v1 = evo_esp[e1].values.astype(float)
            v2 = evo_esp[e2].values.astype(float)
            sim = 1 - cosine(v1, v2)
            cosseno_results.append({
                'par': f'{e1} × {e2}', 'cosseno': round(sim, 4),
            })
    stats['cosseno'] = cosseno_results

    # ── Coeficiente de variação por partido (top 15) ──
    cv_list = []
    for partido in df['Partido'].value_counts().head(15).index:
        serie = (
            df[df['Partido'] == partido]
            .groupby('Ano').size()
            .reindex(year_range, fill_value=0)
        )
        cv = float(serie.std() / serie.mean()) if serie.mean() > 0 else 0
        cv_list.append({
            'partido': partido, 'cv': round(cv, 3), 'total': int(serie.sum()),
        })
    stats['cv_por_partido'] = sorted(cv_list, key=lambda x: x['cv'])

    # ── Top 15 partidos ──
    top15 = df['Partido'].value_counts().head(15)
    stats['top_partidos'] = [
        {'partido': k, 'count': int(v)} for k, v in top15.items()
    ]

    # ── Por região ──
    stats['por_regiao'] = (
        df['Regiao'].value_counts()
        .reset_index()
        .rename(columns={'index': 'regiao', 'Regiao': 'regiao', 'count': 'count'})
        .to_dict('records')
    )

    # ── Por UF ──
    stats['por_uf'] = (
        df['Estado'].value_counts()
        .reset_index()
        .rename(columns={'index': 'uf', 'Estado': 'uf', 'count': 'count'})
        .to_dict('records')
    )

    # ── Por fase da sessão ──
    stats['por_fase'] = (
        df['Fase'].value_counts()
        .reset_index()
        .rename(columns={'index': 'fase', 'Fase': 'fase', 'count': 'count'})
        .to_dict('records')
    )

    # ── Vocabulário por espectro ──
    stats['vocabulario'] = {}
    for esp in espectros:
        textos = df[df['Espectro'] == esp]['Discurso'].dropna()
        stats['vocabulario'][esp] = _top_words(textos)

    # ── Análises por tópico (se disponível) ──
    if topics is not None and topic_labels is not None:
        df_with_topics = df.copy()
        df_with_topics['Topico'] = topics
        df_with_topics['Topico_label'] = df_with_topics['Topico'].map(topic_labels)
        stats.update(_topic_statistics(df_with_topics, topic_labels, year_range))

    return stats


def _top_words(texts: pd.Series, n: int = 25) -> list[dict]:
    """Top N palavras de uma série de textos, excluindo stopwords de domínio."""
    import re
    counter = Counter()
    for t in texts:
        if pd.isna(t):
            continue
        words = re.findall(r'\b[a-záàâãéèêíïóôõúüç]{4,}\b', t.lower())
        counter.update(w for w in words if w not in STOPWORDS_DOMAIN)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]


def _topic_statistics(
    df: pd.DataFrame,
    topic_labels: dict[int, str],
    year_range: range,
) -> dict:
    """Análises cruzadas por tópico: frequência, espectro, ano, região."""
    stats = {}
    valid = df[df['Topico'] != -1]

    # Frequência geral
    freq = (
        valid['Topico_label'].value_counts(normalize=True)
        .mul(100).round(2)
        .reset_index()
    )
    freq.columns = ['topico', 'pct']
    stats['topicos_freq'] = freq.to_dict('records')

    # Tópicos × espectro
    t_esp = []
    for esp in valid['Espectro'].unique():
        mask = valid['Espectro'] == esp
        f = valid.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in f.items():
            t_esp.append({'espectro': esp, 'topico': topico, 'pct': float(pct)})
    stats['topicos_por_espectro'] = t_esp

    # Tópicos × ano
    t_ano = []
    for ano in sorted(valid['Ano'].dropna().unique()):
        mask = valid['Ano'] == ano
        f = valid.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in f.items():
            t_ano.append({'ano': int(ano), 'topico': topico, 'pct': float(pct)})
    stats['topicos_por_ano'] = t_ano

    # Tópicos × região
    t_reg = []
    for reg in valid['Regiao'].unique():
        mask = valid['Regiao'] == reg
        f = valid.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in f.items():
            t_reg.append({'regiao': reg, 'topico': topico, 'pct': float(pct)})
    stats['topicos_por_regiao'] = t_reg

    # Evolução temporal por tópico (novo)
    t_evo = []
    for tid, label in topic_labels.items():
        if tid == -1:
            continue
        serie = (
            valid[valid['Topico'] == tid]
            .groupby('Ano').size()
            .reindex(year_range, fill_value=0)
        )
        for ano, count in serie.items():
            t_evo.append({'topico': label, 'ano': int(ano), 'count': int(count)})
    stats['topicos_evolucao_temporal'] = t_evo

    return stats
