"""
Scraper do SitaQWeb (sistema legado da Câmara dos Deputados).

Busca discursos por keyword diretamente no HTML.
Retorna texto original COM acentos (diferente do scraper legado que normalizava).
"""

import re
import time
import html as html_module
import unicodedata

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.camara.leg.br/internet/sitaqweb/resultadoPesquisaDiscursos.asp"
DETAIL_BASE = "https://www.camara.leg.br/internet/sitaqweb/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
    ),
    "Referer": "https://www.camara.leg.br/",
}


def scrape(
    keywords: list[str],
    data_inicio: str = "01/01/2000",
    data_fim: str = "31/12/2025",
    base_pesq: str = "plenario",
    page_size: int = 50,
    delay: float = 0.2,
    max_pages: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """
    Raspa discursos do SitaQWeb por keywords.

    Args:
        keywords: Lista de termos de busca (ex: ["feminicídio", "violência doméstica"])
        data_inicio: Data inicial DD/MM/YYYY
        data_fim: Data final DD/MM/YYYY
        base_pesq: "plenario" ou "comissoes"
        page_size: Resultados por página (max 50)
        delay: Delay entre requisições (segundos)
        max_pages: Limite de páginas (None = todas)
        progress_callback: Callable(page: int, total_so_far: int)

    Returns:
        Lista de dicts com: Data, Sessao, Orador, Partido, Estado, Fase, Hora, Discurso
    """
    _cb = progress_callback or (lambda *a: None)

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
        "basePesq": base_pesq,
        "CampoOrdenacao": "dtSessao",
        "TipoOrdenacao": "DESC",
        "PageSize": page_size,
    }

    results = []
    page = 1

    while True:
        if max_pages and page > max_pages:
            break

        params["CurrentPage"] = page
        resp = _get(BASE_URL, params=params, timeout=60)
        if resp is None:
            print(f"Desistindo na página {page} após {len(RETRY_WAITS) + 1} tentativas")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table tbody tr")
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue

            record = _parse_row(cols)
            if record:
                results.append(record)

            time.sleep(delay)

        _cb(page, len(results))

        next_link = soup.find("a", string=lambda t: t and "Próxima" in t)
        if not next_link:
            break

        page += 1

    return results


RETRY_WAITS = (5, 15, 45)
RETRY_STATUS = {429, 500, 502, 503, 504}


def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response | None:
    """GET com novas tentativas e espera crescente para erros temporários do servidor."""
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        t0 = time.time()
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            problem = f"HTTP {resp.status_code}"
            if resp.status_code not in RETRY_STATUS:
                print(f"[sitaqweb] {problem} (não temporário), sem nova tentativa: {url[:100]}")
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            problem = type(e).__name__

        elapsed = time.time() - t0
        if wait is None:
            print(f"[sitaqweb] {problem} após {elapsed:.0f}s, tentativa {attempt}: desistindo")
            return None
        print(f"[sitaqweb] {problem} após {elapsed:.0f}s, tentativa {attempt}: nova tentativa em {wait}s")
        time.sleep(wait)


def _parse_row(cols) -> dict | None:
    """Extrai dados de uma linha da tabela de resultados."""
    data = cols[0].get_text(strip=True)
    sessao = cols[1].get_text(strip=True)
    fase = cols[2].get_text(strip=True)
    hora = cols[6].get_text(strip=True) if len(cols) > 6 else ""

    orador_raw = cols[5].get_text(strip=True)
    nome, partido, estado = _parse_orador(orador_raw)

    link_elem = cols[3].find("a")
    if not link_elem or "href" not in link_elem.attrs:
        return None

    url = urljoin(DETAIL_BASE, link_elem["href"].strip())

    r = _get(url, timeout=30)
    discurso = _extract_speech_text(r.text) if r is not None else "[Erro: falha ao baixar discurso]"

    return {
        "Data": data,
        "Sessao": sessao,
        "Orador": nome,
        "Partido": partido,
        "Estado": estado,
        "Fase": fase,
        "Hora": hora,
        "Discurso": discurso,
    }


def _parse_orador(raw: str) -> tuple[str, str, str]:
    """Extrai nome, partido e estado do campo orador."""
    parts = re.split(r"[,\-]", raw)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 3:
        return parts[0], parts[-2], parts[-1]
    elif len(parts) == 2:
        return parts[0], parts[1], ""
    elif len(parts) == 1:
        return parts[0], "SEM PARTIDO", ""
    return "SEM NOME", "SEM PARTIDO", ""


def _extract_speech_text(html_str: str) -> str:
    """Extrai texto do discurso da página de detalhe. Preserva acentos."""
    try:
        html_str = html_str.encode('latin-1').decode('utf-8', errors='ignore')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    soup = BeautifulSoup(html_str, "html.parser")
    hr_tag = soup.find("hr")
    if not hr_tag:
        return ""

    paragraphs = []
    for p in hr_tag.find_all_next("p"):
        # unclosed <p> tags make html.parser nest them; the outer one repeats the whole speech
        if p.find("p"):
            continue
        text = p.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    full_text = " ".join(paragraphs)
    full_text = html_module.unescape(full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text
