"""
Scraper híbrido: SitaQWeb pra descoberta + API pra texto original.

1. SitaQWeb busca por keyword → lista de (orador, partido, data)
2. Filtra deputados únicos que discursaram sobre o tema
3. API puxa discursos desses deputados
4. Filtra por keyword no campo transcricao
"""

import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

from . import api as api_scraper

SITAQWEB_URL = "https://www.camara.leg.br/internet/sitaqweb/resultadoPesquisaDiscursos.asp"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
    ),
    "Referer": "https://www.camara.leg.br/",
}


def scrape(
    keywords: list[str],
    data_inicio_sitaq: str = "01/01/2000",
    data_fim_sitaq: str = "31/12/2025",
    data_inicio_api: str = "2000-01-01",
    data_fim_api: str = "2025-12-31",
    max_workers: int = 5,
    delay: float = 0.3,
    sitaq_max_pages: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """
    Scraper híbrido: SitaQWeb descobre deputados → API puxa texto original.

    Args:
        keywords: Termos de busca
        data_inicio_sitaq: DD/MM/YYYY (formato SitaQWeb)
        data_fim_sitaq: DD/MM/YYYY
        data_inicio_api: YYYY-MM-DD (formato API)
        data_fim_api: YYYY-MM-DD
        max_workers: Requisições paralelas na API
        delay: Delay entre requisições
        sitaq_max_pages: Limite de páginas no SitaQWeb (pra testes)
        progress_callback: Callable(stage: str, detail: str)

    Returns:
        Lista de dicts no formato padrão do pipeline.
    """
    _cb = progress_callback or (lambda *a: None)

    # ── Fase 1: SitaQWeb descobre quem discursou ──
    _cb("sitaqweb", "Buscando oradores no SitaQWeb...")
    oradores = _discover_speakers(
        keywords, data_inicio_sitaq, data_fim_sitaq, sitaq_max_pages, delay,
    )
    print(f"SitaQWeb encontrou {len(oradores)} oradores únicos")
    _cb("sitaqweb", f"{len(oradores)} oradores encontrados")

    # ── Fase 2: Mapeia oradores pra IDs da API ──
    _cb("mapping", "Mapeando oradores para IDs da API...")
    all_deputados = api_scraper._list_deputados()
    dep_by_name = {}
    for dep in all_deputados:
        name = dep.get("nome", "").upper().strip()
        dep_by_name[name] = dep

    matched_deps = []
    unmatched = []
    for orador in oradores:
        name_upper = orador.upper().strip()
        if name_upper in dep_by_name:
            matched_deps.append(dep_by_name[name_upper])
        else:
            for dep_name, dep in dep_by_name.items():
                if name_upper in dep_name or dep_name in name_upper:
                    matched_deps.append(dep)
                    break
            else:
                unmatched.append(orador)

    matched_deps = list({d["id"]: d for d in matched_deps}.values())
    print(f"Mapeados: {len(matched_deps)} deputados, {len(unmatched)} sem match")
    _cb("mapping", f"{len(matched_deps)} deputados mapeados")

    # ── Fase 3: API puxa discursos desses deputados ──
    _cb("api", "Buscando discursos na API...")
    kw_pattern = re.compile(
        "|".join(re.escape(kw) for kw in keywords),
        re.IGNORECASE,
    )

    results = []
    for i, dep in enumerate(matched_deps):
        speeches = api_scraper._get_speeches_for_deputado(
            dep["id"], data_inicio_api, data_fim_api, delay,
        )
        for speech in speeches:
            text = speech.get("transcricao", "") or ""
            if kw_pattern.search(text):
                results.append(api_scraper._normalize_record(speech, dep))

        _cb("api", f"Deputado {i+1}/{len(matched_deps)}, {len(results)} discursos")

    return results


def _discover_speakers(
    keywords: list[str],
    data_inicio: str,
    data_fim: str,
    max_pages: int | None,
    delay: float,
) -> list[str]:
    """Usa SitaQWeb pra descobrir quais oradores discursaram sobre o tema."""

    def _strip_accents(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    tx_texto = " OR ".join(f'"{_strip_accents(kw)}"' for kw in keywords)

    params = {
        "CurrentPage": 1,
        "txOrador": "",
        "txPartido": "",
        "txUF": "",
        "dtInicio": data_inicio,
        "dtFim": data_fim,
        "txTexto": tx_texto,
        "txSumario": "",
        "basePesq": "plenario",
        "CampoOrdenacao": "dtSessao",
        "TipoOrdenacao": "DESC",
        "PageSize": 50,
    }

    oradores = set()
    page = 1

    while True:
        if max_pages and page > max_pages:
            break

        params["CurrentPage"] = page
        try:
            resp = requests.get(SITAQWEB_URL, params=params, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Erro na página {page}: {e}")
            break

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table tbody tr")
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 6:
                orador_raw = cols[5].get_text(strip=True)
                parts = re.split(r"[,\-]", orador_raw)
                if parts:
                    oradores.add(parts[0].strip())

        next_link = soup.find("a", string=lambda t: t and "Próxima" in t)
        if not next_link:
            break

        page += 1
        time.sleep(delay)

    return list(oradores)
