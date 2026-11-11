"""
Bancada de cada partido na posse de cada legislatura, via API Dados Abertos.

A API não tem essa contagem pronta: `/deputados?idLegislatura=N` traz uma linha por partido
pelo qual o deputado passou na legislatura (sem data) e inclui suplentes. Por isso o histórico
de cada deputado (`/deputados/{id}/historico`) é consultado, e conta-se quem estava em
exercício no fim do dia da posse (1º de fevereiro), com o partido daquele momento.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests

API_BASE = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json"}
WORKERS = 5
TIMEOUT = 30
RETRIES = 3


class BancadaError(Exception):
    pass


def legislaturas(inicio: date, fim: date) -> list[dict]:
    """Legislaturas que cruzam o período: [{'id', 'inicio', 'fim'}] (datas ISO)."""
    out = []
    for leg in _get_all(f"{API_BASE}/legislaturas", {"ordem": "ASC", "ordenarPor": "id"}):
        ini, end = date.fromisoformat(leg["dataInicio"]), date.fromisoformat(leg["dataFim"])
        if ini <= fim and end >= inicio:
            out.append({"id": leg["id"], "inicio": leg["dataInicio"], "fim": leg["dataFim"]})
    return out


def posse(legislatura: int, data_inicio: str, progress_callback=None) -> tuple[dict[str, int], str]:
    """
    ({sigla: deputados em exercício no fim do dia da posse}, fonte). Levanta BancadaError se algum
    histórico não puder ser baixado (uma contagem incompleta não deve ir para o cache).

    fonte = 'historico' (posse exata) ou 'listagem': a API não tem histórico das legislaturas
    antigas (até a 51, 1999–2003); aí conta-se cada nome da listagem uma vez, com o partido
    que ela mostra. A listagem inclui suplentes (638 nomes na 51), então só as fatias de cada
    espectro valem, não os números absolutos; na 51 elas ficam a ~1 ponto do resultado de 1998.
    """
    listing = _get_all(f"{API_BASE}/deputados", {"idLegislatura": legislatura})
    ids = sorted({d["id"] for d in listing})
    cutoff = datetime.fromisoformat(data_inicio) + timedelta(days=1)  # posse is on the first day
    cb = progress_callback or (lambda *a: None)

    counts: dict[str, int] = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, sigla in enumerate(pool.map(lambda d: _party_at(d, legislatura, cutoff), ids), start=1):
            if sigla is _FAILED:
                failed += 1
            elif sigla:
                counts[sigla] = counts.get(sigla, 0) + 1
            cb(i, len(ids))
    if failed:
        raise BancadaError(f"{failed} históricos de deputados da legislatura {legislatura} não foram baixados")
    if counts:
        return counts, "historico"

    seen: set[int] = set()
    for d in listing:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        # old parties that later reused the acronym are marked with '*' ('PL*', 'PSD*')
        sigla = (d.get("siglaPartido") or "SEM PARTIDO").rstrip("*")
        counts[sigla] = counts.get(sigla, 0) + 1
    return counts, "listagem"


_FAILED = object()


def _party_at(deputado: int, legislatura: int, cutoff: datetime):
    """Partido do deputado em `cutoff` se estava em exercício; None se não; _FAILED se a consulta falhou."""
    try:
        events = _get(f"{API_BASE}/deputados/{deputado}/historico", {})["dados"]
    except BancadaError:
        return _FAILED
    events = sorted(
        (e for e in events if e.get("idLegislatura") == legislatura and e.get("dataHora")),
        key=lambda e: e["dataHora"],
    )
    current = None
    for e in events:
        if datetime.fromisoformat(e["dataHora"]) > cutoff:
            break
        current = e
    if current and current.get("situacao") == "Exercício":
        return current.get("siglaPartido") or "SEM PARTIDO"
    return None


def _get_all(url: str, params: dict) -> list[dict]:
    rows, page = [], 1
    while True:
        data = _get(url, {**params, "itens": 100, "pagina": page})
        rows += data["dados"]
        if not any(link.get("rel") == "next" for link in data.get("links", [])):
            return rows
        page += 1


def _get(url: str, params: dict) -> dict:
    last = None
    for _ in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
    raise BancadaError(f"falha ao consultar {url}: {last}")
