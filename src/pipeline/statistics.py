"""
Análises estatísticas: tendências (Mann-Kendall), Pearson, cosseno, CV, e análises por tópico.
"""

import re
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from .config import STOPWORDS_DOMAIN
from .trends import MIN_YEARS_ANNUAL, complete_years, compute_trends, series_start
from .vocabulary import fold, ubiquitous_words


def compute_all_statistics(
    df: pd.DataFrame,
    topics: np.ndarray | None = None,
    topic_labels: dict[int, str] | None = None,
    period: tuple[date, date] | None = None,
    plenary_totals: dict[str, int | None] | None = None,
    bancadas: list[dict] | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """
    Calcula todas as análises estatísticas.

    Args:
        df: DataFrame com colunas Data_dt, Ano, Espectro, Partido, Regiao, Discurso.
        topics: Array de tópico dominante por documento (mesmo índice que df).
        topic_labels: Mapeamento topic_id → label.
        period: período coletado; None = da primeira à última data dos discursos.
        plenary_totals: {'YYYY-MM': total do Plenário} para as tendências em proporção.
        bancadas: bancada na posse de cada legislatura, para a "proporção ÷ bancada" dos espectros.
        keywords: keywords da busca (tabela keyword × tópico, com 2+ keywords).

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

    # ── Tendências (Mann-Kendall anual e sazonal, contagem e proporção) ──
    inicio, fim = period or (df['Data_dt'].min().date(), df['Data_dt'].max().date())
    if topics is not None:
        df = df.assign(Topico=topics)
    stats['tendencias'] = compute_trends(df, inicio, fim, plenary_totals, topic_labels, bancadas)
    espectros = ['Esquerda', 'Centro', 'Direita']

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

    # ── Correlação de Pearson entre espectros (anos completos) ──
    # same start as the trends: years before the first speech would be zeros for every spectrum
    # and inflate the correlation
    pearson_results = []
    years = complete_years(series_start(df, inicio), fim)
    evo_esp = evo_esp.reindex(years, fill_value=0)
    cols = [c for c in espectros if c in evo_esp.columns] if len(years) >= MIN_YEARS_ANNUAL else []
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
    # with a single year there is no variation to measure
    partidos_cv = df['Partido'].value_counts().head(15).index if len(year_range) >= 2 else []
    for partido in partidos_cv:
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
    col = 'Texto_modelo' if 'Texto_modelo' in df.columns else 'Discurso'  # texto limpo, se houver
    excluded = STOPWORDS_DOMAIN | ubiquitous_words(df[col].dropna().tolist(), STOPWORDS_DOMAIN)
    for esp in espectros:
        textos = df[df['Espectro'] == esp][col].dropna()
        stats['vocabulario'][esp] = _top_words(textos, excluded)

    # ── Análises por tópico (se disponível) ──
    if topics is not None and topic_labels is not None:
        df_with_topics = df.copy()
        df_with_topics['Topico'] = topics
        df_with_topics['Topico_label'] = df_with_topics['Topico'].map(topic_labels)
        stats.update(_topic_statistics(df_with_topics, topic_labels, year_range))
        stats['keywords_por_topico'] = keywords_by_topic(df_with_topics, keywords, topic_labels)
        stats['mencoes_por_topico'] = mentions_by_topic(df_with_topics, keywords, topic_labels)

    return stats


def _top_words(texts: pd.Series, excluded: set[str], n: int = 25) -> list[dict]:
    """Top N palavras de uma série de textos, sem as palavras em excluded."""
    counter = Counter()
    for t in texts:
        if pd.isna(t):
            continue
        words = re.findall(r'\b[a-záàâãéèêíïóôõúüç]{4,}\b', t.lower())
        counter.update(w for w in words if w not in excluded)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]


def keyword_pattern(keyword: str) -> re.Pattern:
    """Regex de uma keyword no texto já passado por fold(): sem acento, plural opcional, espaços flexíveis."""
    words = fold(keyword).split()
    body = r'\s+'.join(re.escape(w) + r'(?:e?s)?' for w in words)
    return re.compile(rf'(?<!\w){body}(?!\w)')


def keywords_by_topic(df: pd.DataFrame, keywords: list[str], topic_labels: dict[int, str]) -> dict | None:
    """
    % dos discursos de cada tópico (principal) que contêm cada keyword, no texto original.

    Só faz sentido com 2+ keywords: com uma, ela está em todos os discursos.
    """
    keywords = [k for k in dict.fromkeys(k.strip() for k in keywords or []) if k]
    if len(keywords) < 2 or 'Discurso' not in df.columns:
        return None
    texts = df['Discurso'].fillna('').map(fold)
    has = pd.DataFrame({k: texts.str.contains(keyword_pattern(k)) for k in keywords}, index=df.index)

    def pcts(mask) -> dict:
        return {k: round(float(has.loc[mask, k].mean() * 100), 1) for k in keywords}

    rows = []
    for tid, n in df.loc[df['Topico'] != -1, 'Topico'].value_counts().items():
        rows.append({'topico': topic_labels.get(tid, str(tid)), 'n': int(n), 'pct': pcts(df['Topico'] == tid)})
    return {'keywords': keywords, 'geral': pcts(slice(None)), 'topicos': rows}


def mentions_by_topic(df: pd.DataFrame, keywords: list[str], topic_labels: dict[int, str]) -> dict | None:
    """
    Quantas vezes cada discurso cita as keywords da busca, resumido por tópico (principal).

    Discursos que citam o termo uma vez só costumam ser sobre outro assunto (homenagens,
    posses, eleições) e formam tópicos próprios; a contagem mostra quais são. Conta no texto
    limpo (sem cabeçalho nem falas de outros oradores), sem diferenciar acento e plural, e soma
    todas as keywords. Nenhum discurso é filtrado: um discurso sobre o tema pode citar o termo
    uma vez e usar outras palavras ("violência contra a mulher", "Maria da Penha").
    """
    keywords = [k for k in dict.fromkeys(k.strip() for k in keywords or []) if k]
    col = 'Texto_modelo' if 'Texto_modelo' in df.columns else 'Discurso'
    if not keywords or col not in df.columns:
        return None
    texts = df[col].fillna('').map(fold)
    counts = sum(texts.str.count(keyword_pattern(k)) for k in keywords)

    def summary(c: pd.Series) -> dict:
        return {'n': int(len(c)), 'mediana': float(c.median()), 'pct_uma': round(float((c == 1).mean() * 100), 1)}

    valid = df['Topico'] != -1
    rows = [{'topico': topic_labels.get(tid, str(tid)), **summary(counts[df['Topico'] == tid])}
            for tid in df.loc[valid, 'Topico'].unique()]
    return {'keywords': keywords, 'geral': summary(counts), 'topicos': rows}


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

    # Perfil de tópicos dos 15 partidos com mais discursos (% dos discursos do partido)
    top_topicos = valid['Topico_label'].value_counts().head(8).index.tolist()
    perfil = []
    for partido, grupo in valid.groupby('Partido'):
        dist = grupo['Topico_label'].value_counts(normalize=True).mul(100)
        perfil.append({
            'partido': partido,
            'espectro': grupo['Espectro'].mode().iat[0],
            'total': len(grupo),
            'topicos': {t: round(float(dist.get(t, 0.0)), 1) for t in top_topicos},
        })
    stats['partidos_topicos'] = sorted(perfil, key=lambda x: -x['total'])[:15]

    return stats
