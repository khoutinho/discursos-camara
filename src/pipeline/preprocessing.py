"""
Carregamento, limpeza e classificação dos dados.
"""

import pandas as pd

from .config import ESPECTRO, ESPECTRO_3, PARTIDO_GRAFIAS, UF_REGIAO

_SIGLAS = {k.upper(): k for k in [*ESPECTRO, *PARTIDO_GRAFIAS.values()]} | {
    k.upper(): v for k, v in PARTIDO_GRAFIAS.items()
}


def load_and_clean(csv_path: str, encoding: str = 'utf-8-sig') -> pd.DataFrame:
    """Carrega CSV de discursos e faz limpeza básica (datas, colunas derivadas)."""
    return clean(pd.read_csv(csv_path, encoding=encoding))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona Data_dt, Ano e Mes; descarta linhas sem data válida."""
    df = df.copy()
    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['Ano'] = df['Data_dt'].dt.year.astype('Int64')
    df['Mes'] = df['Data_dt'].dt.month.astype('Int64')

    df = df.dropna(subset=['Data_dt']).copy()
    return df


def normalize_party(sigla):
    """Une grafias da mesma sigla ('PCDOB' e 'PCdoB', 'S.PART.' e 'SEM PARTIDO'); o resto fica como está."""
    if not isinstance(sigla, str):
        return sigla
    sigla = sigla.strip()
    return _SIGLAS.get(sigla.upper(), sigla)


def spectrum_of(sigla) -> str:
    """Espectro em 3 classes de uma sigla ('Sem classificação' se não estiver no mapa)."""
    return ESPECTRO_3.get(ESPECTRO.get(normalize_party(sigla)), 'Sem classificação')


def classify_ideology(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a coluna Partido e adiciona Espectro7 e Espectro (3 classes)."""
    df = df.copy()
    df['Partido'] = df['Partido'].map(normalize_party)
    df['Espectro7'] = df['Partido'].map(ESPECTRO).fillna('Sem classificação')
    df['Espectro'] = df['Espectro7'].map(ESPECTRO_3).fillna('Sem classificação')
    return df


def classify_region(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna Regiao ao DataFrame."""
    df = df.copy()
    df['Regiao'] = df['Estado'].map(UF_REGIAO).fillna('Não identificado')
    return df
