#!/usr/bin/env python3
"""
Pipeline NLP - Discursos da Câmara dos Deputados sobre Violência contra a Mulher
Metodologia baseada em: "Mudanças Climáticas no Discurso Parlamentar Brasileiro" (2025)

Etapas:
1. Carregamento e limpeza dos dados
2. Classificação ideológica dos partidos (Bolognesi et al., 2023)
3. Classificação geográfica por região
4. Análise estatística e comparação de séries temporais
   - Teste de Mann-Kendall
   - Similaridade do cosseno
   - Coeficiente de correlação de Pearson
   - Coeficiente de variação
5. Modelagem de tópicos (BERTopic + k-means com busca em grade)
6. Análise dos tópicos por ideologia, região e tempo
7. Exportação dos dados para dashboard (JSON)
"""

# ─── Instalações necessárias (rode uma vez antes) ────────────────────────────
# pip install pandas numpy scipy pymannkendall scikit-learn
# pip install bertopic sentence-transformers spacy nltk
# python -m spacy download pt_core_news_sm
# python -m nltk.downloader stopwords punkt

import os
import re
import json
import warnings
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import pymannkendall as mk

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── CONFIGURAÇÃO ────────────────────────────────────────────────────────────

INPUT_CSV  = r'C:\Users\anton\Downloads\PROJETO-CAMARA\discursos_completo.csv'
OUTPUT_DIR = r'C:\Users\anton\Downloads\PROJETO-CAMARA'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. CARREGAMENTO E LIMPEZA ───────────────────────────────────────────────
print("=" * 60)
print("1. CARREGAMENTO E LIMPEZA DOS DADOS")
print("=" * 60)

df = pd.read_csv(INPUT_CSV, encoding='utf-8')
print(f"Total de registros: {len(df)}")
print(f"Colunas: {list(df.columns)}")

# Parse de data
df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
df['Ano']     = df['Data_dt'].dt.year.astype('Int64')
df['Mes']     = df['Data_dt'].dt.month.astype('Int64')

# Remove registros sem data válida
df_clean = df.dropna(subset=['Data_dt']).copy()
print(f"Após limpeza de datas: {len(df_clean)} registros")
print(f"Com discurso: {df_clean['Discurso'].notna().sum()}")
print(f"Sem discurso: {df_clean['Discurso'].isna().sum()}")


# ─── 2. CLASSIFICAÇÃO IDEOLÓGICA ─────────────────────────────────────────────
# Baseado em Bolognesi, Ribeiro & Codato (2023) + literatura complementar
# Escala 0-10: extrema-esquerda(0-1.5), esquerda(1.51-3), centro-esquerda(3.01-4.49),
#              centro(4.5-5.5), centro-direita(5.51-7), direita(7.01-8.5), extrema-direita(8.51-10)
print("\n" + "=" * 60)
print("2. CLASSIFICAÇÃO IDEOLÓGICA DOS PARTIDOS")
print("=" * 60)

ESPECTRO = {
    # Extrema-esquerda
    'PSOL': 'Extrema-esquerda', 'PCdoB': 'Extrema-esquerda', 'PCB': 'Extrema-esquerda',
    'UP': 'Extrema-esquerda',

    # Esquerda
    'PT': 'Esquerda', 'PDT': 'Esquerda', 'PSB': 'Esquerda',
    'REDE': 'Esquerda', 'PV': 'Esquerda', 'PMN': 'Esquerda',

    # Centro-esquerda
    'PPS': 'Centro-esquerda', 'PROS': 'Centro-esquerda',

    # Centro
    'MDB': 'Centro', 'PSDB': 'Centro', 'CIDADANIA': 'Centro',
    'PMDB': 'Centro', 'AVANTE': 'Centro',

    # Centro-direita
    'PSD': 'Centro-direita', 'PODE': 'Centro-direita',
    'SOLIDARIEDADE': 'Centro-direita', 'PRD': 'Centro-direita',

    # Direita
    'PL': 'Direita', 'PP': 'Direita', 'PROGRESSISTAS': 'Direita',
    'REPUBLICANOS': 'Direita', 'UNIAO': 'Direita', 'UNIÃO': 'Direita',
    'DEM': 'Direita', 'PSC': 'Direita', 'PRB': 'Direita',
    'PTB': 'Direita', 'PR': 'Direita', 'PATRIOTA': 'Direita',
    'PFL': 'Direita',

    # Extrema-direita
    'PSL': 'Extrema-direita', 'NOVO': 'Extrema-direita',
}

# Mapeamento simplificado esquerda/centro/direita (para análises agregadas)
ESPECTRO_3 = {
    'Extrema-esquerda': 'Esquerda',
    'Esquerda': 'Esquerda',
    'Centro-esquerda': 'Esquerda',
    'Centro': 'Centro',
    'Centro-direita': 'Direita',
    'Direita': 'Direita',
    'Extrema-direita': 'Direita',
}

df_clean['Espectro7'] = df_clean['Partido'].map(ESPECTRO).fillna('Sem classificação')
df_clean['Espectro']  = df_clean['Espectro7'].map(ESPECTRO_3).fillna('Sem classificação')

print("\nDistribuição por espectro (7 classes):")
print(df_clean['Espectro7'].value_counts().to_string())
print("\nDistribuição por espectro (3 classes):")
print(df_clean['Espectro'].value_counts().to_string())


# ─── 3. CLASSIFICAÇÃO GEOGRÁFICA ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CLASSIFICAÇÃO GEOGRÁFICA")
print("=" * 60)

UF_REGIAO = {
    'AC': 'Norte',   'AP': 'Norte',   'AM': 'Norte',  'PA': 'Norte',
    'RO': 'Norte',   'RR': 'Norte',   'TO': 'Norte',
    'AL': 'Nordeste','BA': 'Nordeste','CE': 'Nordeste','MA': 'Nordeste',
    'PB': 'Nordeste','PE': 'Nordeste','PI': 'Nordeste','RN': 'Nordeste','SE': 'Nordeste',
    'DF': 'Centro-Oeste','GO': 'Centro-Oeste','MT': 'Centro-Oeste','MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste','SP': 'Sudeste',
    'PR': 'Sul',     'RS': 'Sul',     'SC': 'Sul',
}
df_clean['Regiao'] = df_clean['Estado'].map(UF_REGIAO).fillna('Não identificado')

print("Distribuição por região:")
print(df_clean['Regiao'].value_counts().to_string())


# ─── 4. ANÁLISE ESTATÍSTICA E SÉRIES TEMPORAIS ───────────────────────────────
print("\n" + "=" * 60)
print("4. ANÁLISE ESTATÍSTICA E COMPARAÇÃO DE SÉRIES TEMPORAIS")
print("=" * 60)

# 4a. Frequência anual geral
evo_ano = df_clean.groupby('Ano').size().reset_index(name='Total')
print("\n--- 4a. Evolução temporal geral ---")
print(evo_ano.to_string(index=False))

# 4b. Teste de Mann-Kendall (tendência monotônica na série toda)
print("\n--- 4b. Teste de Mann-Kendall (série completa) ---")
mk_result = mk.original_test(evo_ano['Total'].values)
print(f"  Tendência: {mk_result.trend}")
print(f"  p-valor:   {mk_result.p:.4f}")
print(f"  Tau:       {mk_result.Tau:.4f}")

# 4c. Mann-Kendall por espectro
print("\n--- 4c. Teste de Mann-Kendall por espectro ideológico ---")
for esp in ['Esquerda', 'Centro', 'Direita']:
    serie = (
        df_clean[df_clean['Espectro'] == esp]
        .groupby('Ano').size()
        .reindex(range(int(evo_ano['Ano'].min()), int(evo_ano['Ano'].max()) + 1), fill_value=0)
    )
    if len(serie) > 3:
        r = mk.original_test(serie.values)
        print(f"  {esp}: tendência={r.trend}, p={r.p:.4f}, Tau={r.Tau:.4f}")

# 4d. Proporção por espectro ao longo do tempo
print("\n--- 4d. Proporção anual por espectro ---")
evo_esp = df_clean.groupby(['Ano', 'Espectro']).size().unstack(fill_value=0)
evo_esp_pct = evo_esp.div(evo_esp.sum(axis=1), axis=0) * 100
print(evo_esp_pct.round(1).to_string())

# 4e. Correlação de Pearson entre espectros
print("\n--- 4e. Correlação de Pearson entre espectros ---")
anos_comuns = evo_esp.index
espectros = [c for c in ['Esquerda', 'Centro', 'Direita'] if c in evo_esp.columns]
for i, e1 in enumerate(espectros):
    for e2 in espectros[i+1:]:
        r, p = pearsonr(evo_esp[e1], evo_esp[e2])
        print(f"  {e1} × {e2}: r={r:.3f}, p={p:.4f}")

# 4f. Similaridade do cosseno entre espectros
print("\n--- 4f. Similaridade do cosseno entre espectros ---")
for i, e1 in enumerate(espectros):
    for e2 in espectros[i+1:]:
        v1 = evo_esp[e1].values.astype(float)
        v2 = evo_esp[e2].values.astype(float)
        sim = 1 - cosine(v1, v2)
        print(f"  {e1} × {e2}: cosseno={sim:.4f}")

# 4g. Coeficiente de variação por partido (top 15)
print("\n--- 4g. Coeficiente de variação por partido (top 15 em volume) ---")
top_partidos = df_clean['Partido'].value_counts().head(15).index
for partido in top_partidos:
    serie = (
        df_clean[df_clean['Partido'] == partido]
        .groupby('Ano').size()
        .reindex(range(int(evo_ano['Ano'].min()), int(evo_ano['Ano'].max()) + 1), fill_value=0)
    )
    cv = serie.std() / serie.mean() if serie.mean() > 0 else 0
    print(f"  {partido}: CV={cv:.2f} (total={serie.sum()})")


# ─── 5. MODELAGEM DE TÓPICOS (BERTopic + k-means) ────────────────────────────
print("\n" + "=" * 60)
print("5. MODELAGEM DE TÓPICOS")
print("=" * 60)

# Palavras-chave para filtrar frases relevantes
# ATENÇÃO: sem acentos pois o CSV já passou por remoção de acentos (NFKD + ascii)
KEYWORDS = [
    'violencia contra a mulher', 'violencia contra as mulheres',
    'violencia domestica', 'lei maria da penha', 'maria da penha',
    'feminicidio', 'femicidio', 'igualdade de genero',
    'equidade de genero', 'politicas para mulheres',
    'direitos das mulheres', 'direitos da mulher', 'machismo', 'misoginia',
    'sexismo', 'violencia sexual', 'estupro', 'assedio sexual',
    'empoderamento feminino', 'autonomia feminina', 'discriminacao de genero',
]
KW_PATTERN = '|'.join([re.escape(k) for k in KEYWORDS])

print("Carregando modelo de linguagem spaCy...")
nlp = spacy.load('pt_core_news_sm', disable=['ner', 'parser'])
nlp.max_length = 2_000_000

print("Carregando modelo de embeddings...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

stop_pt = set(stopwords.words('portuguese'))

def normalizar_frase(frase):
    """Normaliza, lematiza e remove stopwords de uma frase."""
    frase = frase.lower()
    frase = re.sub(r'[0-9]', '', frase)
    frase = re.sub(r'[^\w\s]', ' ', frase)
    doc = nlp(frase)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stop_pt and len(token.lemma_) > 2
    ]
    return ' '.join(tokens)

print("Segmentando discursos em frases e filtrando por palavras-chave...")
df_texto = df_clean[df_clean['Discurso'].notna()].copy()

frases_info = []   # (frase_normalizada, idx_discurso, ano, partido, espectro, regiao)
for idx, row in df_texto.iterrows():
    sentencas = sent_tokenize(str(row['Discurso']), language='portuguese')
    for sent in sentencas:
        if re.search(KW_PATTERN, sent.lower()):
            frase_norm = normalizar_frase(sent)
            if len(frase_norm.split()) >= 4:   # descarta frases muito curtas
                frases_info.append({
                    'frase_original': sent,
                    'frase_norm':     frase_norm,
                    'idx_discurso':   idx,
                    'Ano':            row['Ano'],
                    'Partido':        row['Partido'],
                    'Espectro':       row['Espectro'],
                    'Regiao':         row['Regiao'],
                })

df_frases = pd.DataFrame(frases_info)
print(f"Total de frases relevantes extraídas: {len(df_frases)}")

if len(df_frases) == 0:
    print("ATENÇÃO: Nenhuma frase relevante encontrada. Verifique se os discursos passaram pela remoção de acentos.")
    print("Se sim, adicione versões sem acento nas KEYWORDS acima.")
else:
    print("Gerando embeddings das frases (pode demorar alguns minutos)...")
    embeddings = embedder.encode(
        df_frases['frase_norm'].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )

    # ── Busca em grade para k ideal ──
    print("\n--- Busca em grade para k ideal (k-means) ---")
    print(f"{'k':>5} | {'Coerência':>10} | {'Diversidade':>12} | {'Silhueta':>10}")
    print("-" * 45)

    resultados_grade = []
    K_VALUES = range(10, 71, 10)

    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        # Silhueta (proxy de coerência geométrica)
        # Calcula em amostra se corpus for grande
        n_amostras = min(5000, len(embeddings))
        idx_amostras = np.random.choice(len(embeddings), n_amostras, replace=False)
        sil = silhouette_score(embeddings[idx_amostras], labels[idx_amostras])

        # Diversidade: proporção de palavras únicas entre top-10 de cada tópico
        top_words_por_topico = []
        for cluster_id in range(k):
            mask_cluster = labels == cluster_id
            if mask_cluster.sum() == 0:
                continue
            textos_cluster = df_frases.loc[mask_cluster, 'frase_norm']
            all_words = ' '.join(textos_cluster).split()
            top_10 = [w for w, _ in Counter(all_words).most_common(10)]
            top_words_por_topico.append(top_10)

        all_top = [w for lst in top_words_por_topico for w in lst]
        diversidade = len(set(all_top)) / len(all_top) if all_top else 0

        resultados_grade.append({
            'k': k, 'silhueta': round(sil, 4), 'diversidade': round(diversidade, 4)
        })
        print(f"{k:>5} | {sil:>10.4f} | {diversidade:>12.4f} | {sil:>10.4f}")

    df_grade = pd.DataFrame(resultados_grade)

    # Escolhe k com melhor equilíbrio silhueta × diversidade (normaliza e soma)
    for col in ['silhueta', 'diversidade']:
        mn, mx = df_grade[col].min(), df_grade[col].max()
        df_grade[f'{col}_norm'] = (df_grade[col] - mn) / (mx - mn + 1e-9)
    df_grade['score_total'] = df_grade['silhueta_norm'] + df_grade['diversidade_norm']
    k_ideal = int(df_grade.loc[df_grade['score_total'].idxmax(), 'k'])
    print(f"\nk ideal (automático): {k_ideal}")
    print("Avalie os resultados acima e ajuste K_FINAL se preferir outro valor.")

    K_FINAL = k_ideal   # ← altere manualmente se quiser

    # ── Treina BERTopic com k-means ──
    print(f"\nTreinando BERTopic com k-means (k={K_FINAL})...")
    km_final   = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
    ctfidf     = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        hdbscan_model=km_final,
        ctfidf_model=ctfidf,
        language='multilingual',
        verbose=True,
        min_topic_size=5,
    )
    topics, probs = topic_model.fit_transform(
        df_frases['frase_norm'].tolist(), embeddings
    )
    df_frases['Topico'] = topics

    # Nomes dos tópicos (top-5 palavras)
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1:
            topic_labels[tid] = 'Ruído / Genérico'
        else:
            words = topic_model.get_topic(tid)
            if words:
                label = ' | '.join([w for w, _ in words[:5]])
                topic_labels[tid] = f"T{tid}: {label}"

    df_frases['Topico_label'] = df_frases['Topico'].map(topic_labels)

    print("\nTop tópicos por frequência:")
    print(df_frases['Topico_label'].value_counts().head(15).to_string())


# ─── 6. ANÁLISES CRUZADAS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ANÁLISES CRUZADAS")
print("=" * 60)

if len(df_frases) > 0 and 'Topico' in df_frases.columns:
    # Tópicos por espectro
    print("\n--- Proporção de tópicos por espectro ideológico ---")
    topico_esp = (
        df_frases[df_frases['Topico'] != -1]
        .groupby(['Espectro', 'Topico_label'])
        .size()
        .unstack(fill_value=0)
    )
    topico_esp_pct = topico_esp.div(topico_esp.sum(axis=1), axis=0) * 100
    print(topico_esp_pct.round(1).to_string())

    # Tópicos por ano
    print("\n--- Proporção de tópicos por ano ---")
    topico_ano = (
        df_frases[df_frases['Topico'] != -1]
        .groupby(['Ano', 'Topico_label'])
        .size()
        .unstack(fill_value=0)
    )
    topico_ano_pct = topico_ano.div(topico_ano.sum(axis=1), axis=0) * 100
    print(topico_ano_pct.round(1).to_string())

    # Tópicos por região
    print("\n--- Proporção de tópicos por região ---")
    topico_reg = (
        df_frases[df_frases['Topico'] != -1]
        .groupby(['Regiao', 'Topico_label'])
        .size()
        .unstack(fill_value=0)
    )
    topico_reg_pct = topico_reg.div(topico_reg.sum(axis=1), axis=0) * 100
    print(topico_reg_pct.round(1).to_string())


# ─── 7. EXPORTAÇÃO PARA DASHBOARD ────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. EXPORTANDO DADOS PARA DASHBOARD")
print("=" * 60)

dashboard = {}

# Evolução temporal geral
dashboard['evolucao_temporal'] = evo_ano.to_dict('records')

# Espectro por ano
evo_esp_reset = df_clean.groupby(['Ano', 'Espectro']).size().reset_index(name='count')
dashboard['espectro_por_ano'] = evo_esp_reset.to_dict('records')

# Proporção espectro por ano
evo_esp_pct_reset = evo_esp_pct.reset_index().melt(id_vars='Ano', var_name='espectro', value_name='pct')
dashboard['espectro_pct_por_ano'] = evo_esp_pct_reset.dropna().to_dict('records')

# Top 15 partidos
top15 = df_clean['Partido'].value_counts().head(15)
dashboard['top_partidos'] = [{'partido': k, 'count': int(v)} for k, v in top15.items()]

# Partidos por ano (normalizado)
part_ano = df_clean.groupby(['Ano', 'Partido']).size().unstack(fill_value=0)
part_ano_norm = part_ano.div(part_ano.sum(axis=1), axis=0)
dashboard['partidos_por_ano_norm'] = part_ano_norm.reset_index().to_dict('records')

# Por região
_tmp = df_clean['Regiao'].value_counts().reset_index()
_tmp.columns = ['regiao', 'count']
dashboard['por_regiao'] = _tmp.to_dict('records')

# Por UF
_tmp = df_clean['Estado'].value_counts().reset_index()
_tmp.columns = ['uf', 'count']
dashboard['por_uf'] = _tmp.to_dict('records')

# Por fase da sessão
_tmp = df_clean['Fase'].value_counts().reset_index()
_tmp.columns = ['fase', 'count']
dashboard['por_fase'] = _tmp.to_dict('records')

# Coeficiente de variação por partido (para o dashboard)
cv_list = []
for partido in df_clean['Partido'].value_counts().head(15).index:
    serie = (
        df_clean[df_clean['Partido'] == partido]
        .groupby('Ano').size()
        .reindex(range(int(evo_ano['Ano'].min()), int(evo_ano['Ano'].max()) + 1), fill_value=0)
    )
    cv = float(serie.std() / serie.mean()) if serie.mean() > 0 else 0
    cv_list.append({'partido': partido, 'cv': round(cv, 3), 'total': int(serie.sum())})
dashboard['cv_por_partido'] = sorted(cv_list, key=lambda x: x['cv'])

# Mann-Kendall results
mk_results_list = []
for esp in ['Esquerda', 'Centro', 'Direita']:
    serie = (
        df_clean[df_clean['Espectro'] == esp]
        .groupby('Ano').size()
        .reindex(range(int(evo_ano['Ano'].min()), int(evo_ano['Ano'].max()) + 1), fill_value=0)
    )
    if len(serie) > 3:
        r = mk.original_test(serie.values)
        mk_results_list.append({
            'espectro': esp, 'tendencia': r.trend,
            'p': round(r.p, 4), 'tau': round(r.Tau, 4)
        })
dashboard['mann_kendall'] = mk_results_list

# Busca em grade (tabela)
if 'df_grade' in dir():
    dashboard['busca_grade_k'] = df_grade.to_dict('records')

# Tópicos
if len(df_frases) > 0 and 'Topico' in df_frases.columns:
    # Frequência geral
    topicos_freq = (
        df_frases[df_frases['Topico'] != -1]['Topico_label']
        .value_counts(normalize=True)
        .mul(100).round(2)
        .reset_index()
    )
    topicos_freq.columns = ['topico', 'pct']
    dashboard['topicos_freq'] = topicos_freq.to_dict('records')

    # Tópicos × espectro
    t_esp = []
    for esp in df_frases['Espectro'].unique():
        mask = (df_frases['Espectro'] == esp) & (df_frases['Topico'] != -1)
        freq = df_frases.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in freq.items():
            t_esp.append({'espectro': esp, 'topico': topico, 'pct': float(pct)})
    dashboard['topicos_por_espectro'] = t_esp

    # Tópicos × ano
    t_ano = []
    for ano in sorted(df_frases['Ano'].dropna().unique()):
        mask = (df_frases['Ano'] == ano) & (df_frases['Topico'] != -1)
        freq = df_frases.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in freq.items():
            t_ano.append({'ano': int(ano), 'topico': topico, 'pct': float(pct)})
    dashboard['topicos_por_ano'] = t_ano

    # Tópicos × região
    t_reg = []
    for reg in df_frases['Regiao'].unique():
        mask = (df_frases['Regiao'] == reg) & (df_frases['Topico'] != -1)
        freq = df_frases.loc[mask, 'Topico_label'].value_counts(normalize=True).mul(100).round(2)
        for topico, pct in freq.items():
            t_reg.append({'regiao': reg, 'topico': topico, 'pct': float(pct)})
    dashboard['topicos_por_regiao'] = t_reg

# ─── TEMAS (detecção por regex, sem acentos) ─────────────────────────────────
# Padrões sem acento para bater com o texto já normalizado do CSV
TEMAS = {
    'Violencia domestica': [
        r'violencia domestica', r'violencia contra (?:a |as )?mulher',
        r'lei maria da penha', r'maria da penha', r'agressao domestica',
    ],
    'Feminicidio': [
        r'feminicidio', r'femicidio', r'assassinato de mulher', r'morte de mulher',
    ],
    'Violencia sexual': [
        r'violencia sexual', r'estupro', r'assedio sexual',
        r'abuso sexual', r'exploracao sexual',
    ],
    'Igualdade de genero': [
        r'igualdade de genero', r'equidade de genero',
        r'igualdade salarial', r'discriminacao de genero', r'desigualdade de genero',
    ],
    'Machismo': [
        r'machismo', r'machista', r'misoginia', r'misogino',
        r'sexismo', r'sexista', r'patriarcado',
    ],
    'Protecao das mulheres': [
        r'protecao da(?:s)? mulher', r'delegacia da mulher',
        r'medida protetiva', r'casa da mulher', r'rede de protecao',
    ],
    'Saude da mulher': [
        r'saude da mulher', r'saude materna', r'pre.?natal',
    ],
    'Politicas para mulheres': [
        r'politic(?:a|as) para mulher', r'direitos da(?:s)? mulher',
        r'secretaria da mulher', r'agenda feminina',
    ],
    'Empoderamento feminino': [
        r'empoderamento feminino', r'autonomia feminina',
        r'representacao feminina', r'participacao feminina',
    ],
}

df_com_texto = df_clean[df_clean['Discurso'].notna()].copy()

# Aplica detecção de temas
tema_cols = []
for tema, patterns in TEMAS.items():
    col = f'tema_{tema}'
    combined = '|'.join(patterns)
    df_com_texto[col] = df_com_texto['Discurso'].str.lower().str.contains(
        combined, regex=True, na=False
    ).astype(int)
    tema_cols.append(col)

tema_names = [c.replace('tema_', '') for c in tema_cols]

print("\n--- Frequência de temas ---")
for col, name in zip(tema_cols, tema_names):
    n = df_com_texto[col].sum()
    print(f"  {name}: {n} ({n/len(df_com_texto)*100:.1f}%)")

# Temas por espectro
temas_por_espectro_df = df_com_texto.groupby('Espectro')[tema_cols].mean() * 100
temas_por_espectro_df.columns = tema_names
temas_por_espectro_export = []
for esp in temas_por_espectro_df.index:
    for tema in tema_names:
        temas_por_espectro_export.append({
            'espectro': esp, 'tema': tema,
            'pct': round(float(temas_por_espectro_df.loc[esp, tema]), 1)
        })
dashboard['temas_por_espectro'] = temas_por_espectro_export

# Temas por ano
temas_por_ano_df = df_com_texto.groupby('Ano')[tema_cols].mean() * 100
temas_por_ano_df.columns = tema_names
temas_por_ano_export = []
for ano in temas_por_ano_df.index:
    for tema in tema_names:
        temas_por_ano_export.append({
            'ano': int(ano), 'tema': tema,
            'pct': round(float(temas_por_ano_df.loc[ano, tema]), 1)
        })
dashboard['temas_por_ano'] = temas_por_ano_export

# Temas por região
temas_por_regiao_df = df_com_texto.groupby('Regiao')[tema_cols].mean() * 100
temas_por_regiao_df.columns = tema_names
temas_por_regiao_export = []
for reg in temas_por_regiao_df.index:
    for tema in tema_names:
        temas_por_regiao_export.append({
            'regiao': reg, 'tema': tema,
            'pct': round(float(temas_por_regiao_df.loc[reg, tema]), 1)
        })
dashboard['temas_por_regiao'] = temas_por_regiao_export

# ─── PARTIDOS_TEMAS ───────────────────────────────────────────────────────────
partidos_temas = []
for partido in df_com_texto['Partido'].value_counts().index:
    mask = df_com_texto['Partido'] == partido
    if mask.sum() < 5:
        continue
    row = {
        'partido': partido,
        'espectro': ESPECTRO_3.get(ESPECTRO.get(partido, ''), 'Outro'),
        'total': int(mask.sum()),
    }
    for col, name in zip(tema_cols, tema_names):
        row[name] = round(float(df_com_texto.loc[mask, col].mean() * 100), 1)
    partidos_temas.append(row)
partidos_temas.sort(key=lambda x: x['total'], reverse=True)
dashboard['partidos_temas'] = partidos_temas[:20]

# ─── VOCABULÁRIO POR ESPECTRO ─────────────────────────────────────────────────
import string

STOPWORDS_EXTRA = set("""
a as o os um uma uns umas de do da dos das em no na nos nas por para com
sem sob sobre ao aos entre ate apos durante ante perante ja que nao
mais muito bem tambem como mas ainda quando isso isto ele ela eles elas
eu tu voce voces se si mesmo mesma aqui ali la onde aquele aquela
esse essa este esta nesse nessa neste nesta desse dessa deste desta
ser estar ter haver fazer ir poder dever foi era sera seria
tem tinha tera teria ha havia havera haveria pode podia poderia
e sao eram foram serao seriam estao politicas politica pmdb
aquilo qual quais quem cujo cuja cujos cujas todo toda todos todas
cada outro outra outros outras mesmo mesma mesmos mesmas proprio
entao pois porem porque portanto porquanto contudo todavia
agora depois antes hoje sempre nunca quando pouco
grande pequeno maior menor melhor pior primeiro segundo terceiro
nosso nossa nossos nossas seu sua seus suas meu minha meus minhas
parte numero ano dia vez casa forma tempo governo estado pais
presidente deputado deputada senhor senhora orador oradora camara
plenario sessao discurso brasil brasileiro brasileira congresso
sr sra srs sras voto projeto lei artigo paragrafo inciso
falar dizer disse digo diz deputados contra quero
bloco lider partido revisao
""".split())

stop_vocab = set(stopwords.words('portuguese')) | STOPWORDS_EXTRA

def top_words(texts, n=25):
    counter = Counter()
    for t in texts:
        if pd.isna(t): continue
        words = re.findall(r'\b[a-z]{4,}\b', t.lower())
        counter.update(w for w in words if w not in stop_vocab)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]

vocab = {}
for esp in ['Esquerda', 'Centro', 'Direita']:
    textos = df_com_texto[df_com_texto['Espectro'] == esp]['Discurso']
    vocab[esp] = top_words(textos)
dashboard['vocabulario'] = vocab

json_out = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(dashboard, f, ensure_ascii=False, default=str, indent=2)
print(f"JSON exportado para: {json_out}")

# CSV de frases com tópicos (útil para análise qualitativa)
if len(df_frases) > 0 and 'Topico' in df_frases.columns:
    frases_out = os.path.join(OUTPUT_DIR, 'frases_topicos.csv')
    df_frases[['frase_original', 'Topico', 'Topico_label', 'Ano', 'Partido', 'Espectro', 'Regiao']].to_csv(
        frases_out, index=False, encoding='utf-8'
    )
    print(f"Frases com tópicos: {frases_out}")

# CSV processado completo
csv_out = os.path.join(OUTPUT_DIR, 'discursos_processados.csv')
df_clean.to_csv(csv_out, index=False, encoding='utf-8')
print(f"CSV processado: {csv_out}")


# ─── 8. RESUMO FINAL ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. RESUMO FINAL")
print("=" * 60)
total = len(df_clean)
print(f"Total de discursos limpos:  {total}")
print(f"Com texto:                  {df_clean['Discurso'].notna().sum()}")
print(f"Período:                    {df_clean['Ano'].min()} – {df_clean['Ano'].max()}")
print(f"Partidos únicos:            {df_clean['Partido'].nunique()}")
print(f"UFs únicas:                 {df_clean['Estado'].nunique()}")
print(f"Oradores únicos:            {df_clean['Orador'].nunique()}")
print("\nDistribuição por espectro (3 classes):")
for esp, cnt in df_clean['Espectro'].value_counts().items():
    print(f"  {esp}: {cnt} ({cnt/total*100:.1f}%)")
if len(df_frases) > 0 and 'Topico' in df_frases.columns:
    print(f"\nTotal de frases relevantes: {len(df_frases)}")
    print(f"Tópicos identificados:      {df_frases['Topico'].nunique() - (1 if -1 in df_frases['Topico'].values else 0)}")

print("\n✅ Pipeline concluído!")