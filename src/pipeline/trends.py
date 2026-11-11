"""
Testes de tendência (Mann-Kendall) sobre contagem e proporção de discursos.

- Contagem: discursos do grupo por mês/ano.
- Proporção: discursos do grupo ÷ total de discursos do Plenário no mesmo mês
  (cancela recesso, anos eleitorais e variações gerais de atividade da Câmara).

- Anual: Mann-Kendall sobre a série de anos completos (precisa de MIN_YEARS_ANNUAL).
  Com 4 pontos o menor p possível é ~0,09, então 4 anos nunca dão resultado significativo.
- Sazonal: Mann-Kendall sazonal (Hirsch & Slack, 1984) sobre a série mensal; compara cada
  mês do calendário só com o mesmo mês de outros anos (precisa de MIN_MONTHS_SEASONAL).

A série começa no mês do primeiro discurso da coleta, não na data pesquisada: se o termo
ainda não existia no início do período, os meses vazios antes dele puxariam o teste para
"crescente". Se o primeiro discurso é de fevereiro, a série começa em janeiro: janeiro é
recesso (quase sem discursos no Plenário), e sem isso uma busca iniciada em 1º de janeiro
perderia o primeiro ano do teste anual. Espectros e tópicos usam o mesmo início (o do corpus
inteiro): depois que o termo existe, um grupo sem discursos é informação, não ausência do termo.
"""

from datetime import date

import numpy as np
import pandas as pd
import pymannkendall as mk

MIN_YEARS_ANNUAL = 5
MIN_MONTHS_SEASONAL = 24
ESPECTROS = ['Esquerda', 'Centro', 'Direita']


def coverage(inicio: date, fim: date) -> dict:
    months = pd.period_range(inicio, fim, freq='M')
    years = complete_years(inicio, fim)
    return {
        'inicio': inicio.isoformat(),
        'fim': fim.isoformat(),
        'meses': len(months),
        'anos_completos': len(years),
        'anual_disponivel': len(years) >= MIN_YEARS_ANNUAL,
        'sazonal_disponivel': len(months) >= MIN_MONTHS_SEASONAL,
        'min_anos_anual': MIN_YEARS_ANNUAL,
        'min_meses_sazonal': MIN_MONTHS_SEASONAL,
    }


def complete_years(inicio: date, fim: date) -> list[int]:
    return [y for y in range(inicio.year, fim.year + 1) if date(y, 1, 1) >= inicio and date(y, 12, 31) <= fim]


def compute_trends(
    df: pd.DataFrame,
    inicio: date,
    fim: date,
    plenary_totals: dict[str, int | None] | None = None,
    topic_labels: dict[int, str] | None = None,
) -> dict:
    """
    Args:
        df: discursos com Data_dt, Espectro e (opcional) Topico.
        inicio, fim: período coletado. A série começa no mês do primeiro discurso, se for
            posterior ao de `inicio`; a partir daí, meses sem discursos contam como zero.
        plenary_totals: {'YYYY-MM': total do Plenário}; None = sem proporção.
        topic_labels: {topic_id: label}; None = sem testes por tópico.
    """
    inicio_pesquisado = inicio
    inicio = series_start(df, inicio)
    months = pd.period_range(inicio, fim, freq='M')
    years = complete_years(inicio, fim)
    month_of = df['Data_dt'].dt.to_period('M')

    den = None
    missing = []
    if plenary_totals:
        den = pd.Series([plenary_totals.get(str(m)) for m in months], index=months, dtype=float)
        missing = [str(m) for m in months if pd.isna(den[m])]

    def tests_for(mask: pd.Series) -> dict:
        num = month_of[mask].value_counts().reindex(months, fill_value=0).astype(float)
        return {
            'contagem': _tests(num, None, months, years),
            'proporcao': _tests(num, den, months, years) if den is not None and den.notna().any() else None,
        }

    series = [{'grupo': 'geral', 'nome': 'Total de discursos', 'total': len(df),
               **tests_for(pd.Series(True, index=df.index))}]
    for esp in ESPECTROS:
        mask = df['Espectro'] == esp
        series.append({'grupo': 'espectro', 'nome': esp, 'total': int(mask.sum()), **tests_for(mask)})
    if topic_labels is not None and 'Topico' in df:
        for tid, label in topic_labels.items():
            if tid == -1:
                continue
            mask = df['Topico'] == tid
            series.append({'grupo': 'topico', 'nome': label, 'total': int(mask.sum()), **tests_for(mask)})

    return {
        'cobertura': {**coverage(inicio, fim), 'inicio_pesquisado': inicio_pesquisado.isoformat(),
                      'primeiro_discurso': df['Data_dt'].min().date().isoformat() if len(df) else None,
                      'proporcao_disponivel': den is not None and den.notna().any(),
                      'meses_sem_total': missing},
        'series': series,
    }


def series_start(df: pd.DataFrame, inicio: date) -> date:
    """
    Dia 1 do mês do primeiro discurso (janeiro, se ele for de fevereiro: recesso), se for
    posterior ao mês de `inicio`; senão `inicio`.
    """
    first = df['Data_dt'].min()
    if pd.isna(first):
        return inicio
    start = date(first.year, 1 if first.month == 2 else first.month, 1)
    return start if start > inicio.replace(day=1) else inicio


def _tests(num: pd.Series, den: pd.Series | None, months: pd.PeriodIndex, years: list[int]) -> dict:
    if den is None:
        monthly = num
        annual = num.groupby(num.index.year).sum()
    else:
        known = den > 0  # NaN (failed lookup) and 0 compare False
        monthly = (num / den).where(known)
        annual = num.where(known).groupby(num.index.year).sum() / den.where(known).groupby(den.index.year).sum()

    result = {'anual': None, 'sazonal': None}
    if len(years) >= MIN_YEARS_ANNUAL:
        values = annual.reindex(years).to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) >= MIN_YEARS_ANNUAL:
            result['anual'] = _summary(mk.original_test(values), len(values))
    if len(months) >= MIN_MONTHS_SEASONAL:
        values = monthly.to_numpy(dtype=float)
        n = int((~np.isnan(values)).sum())
        if n >= MIN_MONTHS_SEASONAL:
            # NaN must stay in place: position i and i+12 have to be the same calendar month
            result['sazonal'] = _summary(mk.seasonal_test(values, period=12), n)
    return result


def _summary(r, n: int) -> dict:
    p = float(r.p)
    return {
        'tendencia': r.trend,
        'p': round(p, 4) if np.isfinite(p) else None,
        'tau': round(float(r.Tau), 4) if np.isfinite(r.Tau) else None,
        'n': n,
    }
