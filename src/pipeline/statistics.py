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

from .config import CONVIDADO, ESPECTRO, PARTIDO_GRAFIAS, STOPWORDS_DOMAIN
from .model_config import FW_MIN_DOCS, FW_TOP_N, FW_Z_MIN, MAP_DECIMALS, NGRAM_RANGE
from .trends import MIN_YEARS_ANNUAL, complete_years, compute_trends, series_start
from .vocabulary import TOKEN_PATTERN, ContentVectorizer, excluded_words, fold, ubiquitous_words


def compute_all_statistics(
    df: pd.DataFrame,
    topics: np.ndarray | None = None,
    topic_labels: dict[int, str] | None = None,
    period: tuple[date, date] | None = None,
    plenary_totals: dict[str, int | None] | None = None,
    bancadas: list[dict] | None = None,
    keywords: list[str] | None = None,
    vocab_excluded: set[str] | None = None,
    coords_2d: np.ndarray | None = None,
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
        vocab_excluded: palavras fora do vocabulário dos rótulos (TopicModeler.fit); None = calcula
            (roda o spaCy em todos os discursos).
        coords_2d: coordenadas do mapa 2D (TopicModeler.project_2d), mesma ordem de df; None = sem mapa.

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

    stats['top_oradores'] = top_speakers(df)

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

    # ── Vocabulário por espectro: mais frequentes e características ──
    stats['vocabulario'] = {}
    col = 'Texto_modelo' if 'Texto_modelo' in df.columns else 'Discurso'  # texto limpo, se houver
    docs = df[col].fillna('').tolist()
    excluded = STOPWORDS_DOMAIN | ubiquitous_words(docs, STOPWORDS_DOMAIN)
    for esp in espectros:
        textos = df[df['Espectro'] == esp][col].dropna()
        stats['vocabulario'][esp] = _top_words(textos, excluded)
    if vocab_excluded is None:
        vocab_excluded = excluded_words(docs)
    stats['vocabulario_caracteristico'] = characteristic_words(df, col, vocab_excluded, espectros)

    # ── Análises por tópico (se disponível) ──
    if topics is not None and topic_labels is not None:
        df_with_topics = df.copy()
        df_with_topics['Topico'] = topics
        df_with_topics['Topico_label'] = df_with_topics['Topico'].map(topic_labels)
        stats.update(_topic_statistics(df_with_topics, topic_labels, year_range))
        stats['keywords_por_topico'] = keywords_by_topic(df_with_topics, keywords, topic_labels)
        stats['mencoes_por_topico'] = mentions_by_topic(df_with_topics, keywords, topic_labels)
        if coords_2d is not None:
            stats['mapa'] = embedding_map(df_with_topics, coords_2d)

    return stats


def top_speakers(df: pd.DataFrame, n: int = 10) -> list[dict] | None:
    """
    Os n deputados com mais discursos (convidados ficam de fora).

    O SitaQWeb grava o nome em maiúsculas até 2022 e com só as iniciais maiúsculas depois
    ("ERIKA KOKAY", "Erika Kokay"): o nome é agrupado sem diferenciar maiúsculas e acentos, e
    mostrado na grafia com minúsculas, se houver; senão, com as iniciais maiúsculas ("Iracema Portella"). Quem trocou de partido aparece com o partido
    mais frequente (empate: o mais recente) e o espectro dele; os outros vão em 'outros_partidos'.
    """
    if 'Orador' not in df.columns:
        return None
    d = df[(df['Partido'] != CONVIDADO) & df['Orador'].notna()].sort_values('Data_dt')
    d = d.assign(_chave=d['Orador'].astype(str).map(lambda s: ' '.join(fold(s).split())))
    d = d[d['_chave'] != '']
    result = []
    for _, g in sorted(d.groupby('_chave'), key=lambda kv: (-len(kv[1]), kv[0]))[:n]:
        nomes = g['Orador'].astype(str).str.strip()
        mixed = nomes[nomes != nomes.str.upper()]
        nome = mixed.value_counts().index[0] if len(mixed) else _title_name(nomes.value_counts().index[0])
        partidos = g['Partido'].fillna('?').value_counts()
        last_seen = {p: i for i, p in enumerate(g['Partido'].fillna('?'))}  # sorted by date
        ordem = sorted(partidos.index, key=lambda p: (-partidos[p], -last_seen[p]))
        principal = ordem[0]
        espectro = g.loc[g['Partido'].fillna('?') == principal, 'Espectro'].iloc[0]
        result.append({
            'orador': nome,
            'partido': principal,
            'uf': g['Estado'].value_counts().index[0] if g['Estado'].notna().any() else '',
            'espectro': espectro,
            'discursos': len(g),
            'outros_partidos': [{'partido': p, 'discursos': int(partidos[p])} for p in ordem[1:]],
        })
    return result


def embedding_map(df: pd.DataFrame, coords: np.ndarray) -> dict:
    """
    Pontos do mapa 2D em colunas (uma lista por campo, um item por discurso): com 10.000 discursos,
    uma lista de objetos repetiria os nomes dos campos 10.000 vezes. Orador, partido e espectro vão
    como índices para as listas 'oradores', 'partidos' e 'espectros'; o tópico como id (o rótulo está
    em 'topicos').
    """
    def codes(col: pd.Series) -> tuple[list[int], list[str]]:
        cat = pd.Categorical(col.fillna('?').astype(str))
        return cat.codes.tolist(), list(cat.categories)

    oradores = df['Orador'].map(lambda s: _display_name(str(s)) if pd.notna(s) else '?') \
        if 'Orador' in df.columns else pd.Series('?', index=df.index)
    orador_idx, orador_list = codes(oradores)
    partido_idx, partido_list = codes(df['Partido'])
    espectro_idx, espectro_list = codes(df['Espectro'])
    labels = df.drop_duplicates('Topico').set_index('Topico')['Topico_label']
    return {
        'x': np.round(coords[:, 0], MAP_DECIMALS).tolist(),
        'y': np.round(coords[:, 1], MAP_DECIMALS).tolist(),
        'topico': df['Topico'].astype(int).tolist(),
        'topicos': {str(int(t)): str(l) for t, l in labels.items()},
        'espectro': espectro_idx, 'espectros': espectro_list,
        'orador': orador_idx, 'oradores': orador_list,
        'partido': partido_idx, 'partidos': partido_list,
        'data': df['Data_dt'].dt.strftime('%d/%m/%Y').tolist(),
    }


def _display_name(nome: str) -> str:
    """Nome só em maiúsculas ganha as iniciais maiúsculas; os outros ficam como estão."""
    nome = nome.strip()
    return _title_name(nome) if nome == nome.upper() else nome


_NAME_PARTICLES = {'da', 'de', 'do', 'das', 'dos', 'e'}


def _title_name(nome: str) -> str:
    """'MARIA DO ROSÁRIO' → 'Maria do Rosário' (para nomes que só aparecem em maiúsculas)."""
    words = nome.lower().split()
    return ' '.join(w if i and w in _NAME_PARTICLES else w[:1].upper() + w[1:] for i, w in enumerate(words))


# Siglas que, nos discursos, são sobretudo palavras comuns ("cidadania", "rede de proteção", "União"):
# ficam no vocabulário. Na coleta "feminicídio", "cidadania" aparece 85 vezes em minúsculas e 90 com
# inicial maiúscula; "Republicanos", 4 e 24 (este sai).
_SIGLAS_PALAVRAS = {
    'agir', 'cidadania', 'missão', 'missao', 'mobiliza', 'novo', 'rede', 'solidariedade', 'união', 'uniao',
    'podemos', 'pode',
}


def _party_words() -> set[str]:
    """Siglas de partido em minúsculas ("psol", "republicanos"), menos as que são palavras comuns."""
    siglas = {s.lower() for s in [*ESPECTRO, *PARTIDO_GRAFIAS, *PARTIDO_GRAFIAS.values()]}
    return {s for s in siglas if re.fullmatch(TOKEN_PATTERN, s)} - _SIGLAS_PALAVRAS


def characteristic_words(df: pd.DataFrame, col: str, excluded: set[str], groups: list[str],
                         n: int = FW_TOP_N) -> dict | None:
    """
    Palavras características de cada espectro: log-odds com prior de Dirichlet informativo
    (Monroe, Colaresi & Quinn, 2008, "Fightin' Words"), cada espectro contra os outros dois.

    Para cada termo w (palavra ou par de palavras), y_i = discursos do espectro i com w, y_j = discursos dos
    outros dois espectros com w, n = soma de y sobre todos os termos. O prior α_w é a contagem de w no corpus
    inteiro (α_0 = soma dos α_w), que puxa os termos raros para a média:

        δ_w = log((y_i + α_w) / (n_i + α_0 − y_i − α_w)) − log((y_j + α_w) / (n_j + α_0 − y_j − α_w))
        z_w = δ_w / √(1/(y_i + α_w) + 1/(y_j + α_w))

    Conta discursos, não ocorrências: um discurso que repete a palavra 17 vezes ("baixada fluminense")
    não a torna característica sozinho. Mesmo vocabulário dos rótulos (sem stopwords, palavras ubíquas
    e palavras que não são de conteúdo), sem siglas de partido (o orador cita o próprio partido).
    """
    d = df[df['Espectro'].isin(groups)]
    if d.empty:
        return None
    vec = ContentVectorizer(
        stop_words=sorted(excluded | _party_words()), token_pattern=TOKEN_PATTERN,
        ngram_range=NGRAM_RANGE, min_df=FW_MIN_DOCS, binary=True,
    )
    try:
        X = vec.fit_transform(d[col].fillna('')).tocsc()
    except ValueError:  # nenhum termo em FW_MIN_DOCS discursos
        return None
    terms = vec.get_feature_names_out()
    esp = d['Espectro'].to_numpy()
    alpha = np.asarray(X.sum(axis=0)).ravel().astype(float)
    a0 = alpha.sum()

    result = {'z_min': FW_Z_MIN, 'espectros': {}}
    for g in groups:
        mine = esp == g
        n_docs, n_rest = int(mine.sum()), int((~mine).sum())
        if not n_docs or not n_rest:
            result['espectros'][g] = []
            continue
        yi = np.asarray(X[mine].sum(axis=0)).ravel()
        yj = np.asarray(X[~mine].sum(axis=0)).ravel()
        ni, nj = yi.sum(), yj.sum()
        delta = (np.log((yi + alpha) / (ni + a0 - yi - alpha))
                 - np.log((yj + alpha) / (nj + a0 - yj - alpha)))
        z = delta / np.sqrt(1 / (yi + alpha) + 1 / (yj + alpha))
        top = [k for k in np.argsort(-z)[:n] if z[k] > 0]
        result['espectros'][g] = [{
            'word': terms[k],
            'z': round(float(z[k]), 2),
            'discursos': int(yi[k]),
            'pct': round(yi[k] / n_docs * 100, 1),          # % dos discursos do espectro
            'pct_outros': round(yj[k] / n_rest * 100, 1),   # % dos discursos dos outros dois
        } for k in top]
    return result


def _top_words(texts: pd.Series, excluded: set[str], n: int = 25) -> list[dict]:
    """Top N palavras de uma série de textos, sem as palavras em excluded."""
    counter = Counter()
    for t in texts:
        if pd.isna(t):
            continue
        words = re.findall(r'\b[a-záàâãéèêíïóôõúüç]{4,}\b', t.lower())
        counter.update(w for w in words if w not in excluded)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]


def _keyword_body(keyword: str) -> str:
    return r'\s+'.join(re.escape(w) + r'(?:e?s)?' for w in fold(keyword).split())


def keyword_pattern(keyword: str) -> re.Pattern:
    """Regex de uma keyword no texto já passado por fold(): sem acento, plural opcional, espaços flexíveis."""
    return re.compile(rf'(?<!\w){_keyword_body(keyword)}(?!\w)')


def any_keyword_pattern(keywords: list[str]) -> re.Pattern:
    """
    Regex de qualquer uma das keywords, para contar cada ocorrência uma vez só: "pais" e "país"
    casam com as mesmas palavras (a contagem não diferencia acento), e "violência doméstica" contém
    "violência". As mais longas vêm primeiro, para a ocorrência contar como a keyword inteira.
    """
    bodies = sorted({_keyword_body(k) for k in keywords}, key=len, reverse=True)
    return re.compile(rf'(?<!\w)(?:{"|".join(bodies)})(?!\w)')


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
    limpo (sem cabeçalho nem falas de outros oradores), sem diferenciar acento e plural, juntando
    todas as keywords: cada ocorrência conta uma vez, mesmo que case com mais de uma keyword.
    Nenhum discurso é filtrado: um discurso sobre o tema pode citar o termo uma vez e usar outras
    palavras ("violência contra a mulher", "Maria da Penha").
    """
    keywords = [k for k in dict.fromkeys(k.strip() for k in keywords or []) if k]
    col = 'Texto_modelo' if 'Texto_modelo' in df.columns else 'Discurso'
    if not keywords or col not in df.columns:
        return None
    texts = df[col].fillna('').map(fold)
    counts = texts.str.count(any_keyword_pattern(keywords))

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
