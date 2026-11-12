"""
Script rápido para extrair estatísticas do corpus para o artigo.
Rode antes de escrever a seção Corpus.
"""

import pandas as pd
import numpy as np
import re

INPUT_CSV = r'C:\Users\anton\Downloads\PROJETO-CAMARA\arquivos\discursos_completo2.csv'

df = pd.read_csv(INPUT_CSV, encoding='utf-8')

# ── Parse de data ──────────────────────────────────────────────────
df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
df['Ano'] = df['Data_dt'].dt.year.astype('Int64')
df_clean = df.dropna(subset=['Data_dt']).copy()

# ── Contagem de palavras ───────────────────────────────────────────
def conta_palavras(texto):
    if pd.isna(texto) or str(texto).strip() == '':
        return 0
    return len(str(texto).split())

df_clean['n_palavras'] = df_clean['Discurso'].apply(conta_palavras)
com_texto = df_clean[df_clean['n_palavras'] > 0]
sem_texto = df_clean[df_clean['n_palavras'] == 0]

# ── Impressão ──────────────────────────────────────────────────────
print("=" * 55)
print("ESTATÍSTICAS DO CORPUS — para o artigo")
print("=" * 55)

print(f"\nTotal de registros brutos:          {len(df):>8,}")
print(f"Total após limpeza de datas:        {len(df_clean):>8,}")
print(f"  Com texto de discurso:            {len(com_texto):>8,}")
print(f"  Sem texto (registro vazio):       {len(sem_texto):>8,}")

print(f"\nPeríodo:                            {int(df_clean['Ano'].min())}–{int(df_clean['Ano'].max())}")
print(f"Partidos únicos:                    {df_clean['Partido'].nunique():>8,}")
print(f"Oradores únicos:                    {df_clean['Orador'].nunique():>8,}")
print(f"UFs únicas:                         {df_clean['Estado'].nunique():>8,}")

print(f"\nTotal de palavras (corpus):         {com_texto['n_palavras'].sum():>8,}")
print(f"Média de palavras por discurso:     {com_texto['n_palavras'].mean():>8.1f}")
print(f"Mediana de palavras por discurso:   {com_texto['n_palavras'].median():>8.1f}")
print(f"Máximo de palavras (um discurso):   {com_texto['n_palavras'].max():>8,}")
print(f"Mínimo de palavras (com texto):     {com_texto[com_texto['n_palavras']>0]['n_palavras'].min():>8,}")

print("\nDiscursos por ano:")
por_ano = df_clean.groupby('Ano').size()
for ano, n in por_ano.items():
    bar = '█' * (n // 50)
    print(f"  {ano}: {n:>5}  {bar}")

print("\nTop 10 partidos por volume:")
for partido, n in df_clean['Partido'].value_counts().head(10).items():
    print(f"  {str(partido):<20} {n:>5}")

print("\nDistribuição por fase da sessão:")
for fase, n in df_clean['Fase'].value_counts().items():
    pct = n / len(df_clean) * 100
    print(f"  {str(fase):<30} {n:>5} ({pct:.1f}%)")

print("\nDistribuição por região:")
UF_REGIAO = {
    'AC':'Norte','AP':'Norte','AM':'Norte','PA':'Norte','RO':'Norte','RR':'Norte','TO':'Norte',
    'AL':'Nordeste','BA':'Nordeste','CE':'Nordeste','MA':'Nordeste',
    'PB':'Nordeste','PE':'Nordeste','PI':'Nordeste','RN':'Nordeste','SE':'Nordeste',
    'DF':'Centro-Oeste','GO':'Centro-Oeste','MT':'Centro-Oeste','MS':'Centro-Oeste',
    'ES':'Sudeste','MG':'Sudeste','RJ':'Sudeste','SP':'Sudeste',
    'PR':'Sul','RS':'Sul','SC':'Sul',
}
df_clean['Regiao'] = df_clean['Estado'].map(UF_REGIAO).fillna('Não identificado')
for reg, n in df_clean['Regiao'].value_counts().items():
    pct = n / len(df_clean) * 100
    print(f"  {str(reg):<20} {n:>5} ({pct:.1f}%)")

print("\n" + "=" * 55)
print("Cole esses números na seção Corpus do artigo.")
print("=" * 55)