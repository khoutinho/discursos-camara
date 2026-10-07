"""
Carregamento, limpeza e classificação dos dados.
"""

import pandas as pd

from .config import ESPECTRO, ESPECTRO_3, UF_REGIAO


def load_and_clean(csv_path: str, encoding: str = 'utf-8-sig') -> pd.DataFrame:
    """Carrega CSV de discursos e faz limpeza básica (datas, colunas derivadas)."""
    df = pd.read_csv(csv_path, encoding=encoding)

    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['Ano'] = df['Data_dt'].dt.year.astype('Int64')
    df['Mes'] = df['Data_dt'].dt.month.astype('Int64')

    df = df.dropna(subset=['Data_dt']).copy()
    return df


def classify_ideology(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas Espectro7 e Espectro (3 classes) ao DataFrame."""
    df = df.copy()
    df['Espectro7'] = df['Partido'].map(ESPECTRO).fillna('Sem classificação')
    df['Espectro'] = df['Espectro7'].map(ESPECTRO_3).fillna('Sem classificação')
    return df


def classify_region(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna Regiao ao DataFrame."""
    df = df.copy()
    df['Regiao'] = df['Estado'].map(UF_REGIAO).fillna('Não identificado')
    return df
