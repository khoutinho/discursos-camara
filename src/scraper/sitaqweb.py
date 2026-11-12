"""
Scraper do SitaQWeb (sistema legado da Câmara dos Deputados).

Busca discursos por keyword diretamente no HTML.
Retorna texto original COM acentos (diferente do scraper legado que normalizava).

Duas etapas separadas, para permitir retomar uma coleta interrompida:
- iter_listing(): percorre as páginas de resultados (metadados + link de cada discurso)
- fetch_speech(): baixa o texto de um discurso
"""

import re
import time
import html as html_module
import unicodedata
from collections.abc import Iterator

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

RETRY_WAITS = (5, 15, 45)
RETRY_STATUS = {429, 500, 502, 503, 504}


class ListingError(Exception):
    """A página de resultados não respondeu nem depois das novas tentativas."""


def scrape(
    keywords: list[str],
    data_inicio: str = "01/01/2000",
    data_fim: str = "31/12/2025",
    page_size: int = 50,
    delay: float = 0.2,
    max_pages: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """
    Raspa discursos do Plenário no SitaQWeb por keywords, de uma vez só (sem persistência).

    Args:
        keywords: Lista de termos de busca (ex: ["feminicídio", "violência doméstica"])
        data_inicio: Data inicial DD/MM/YYYY
        data_fim: Data final DD/MM/YYYY
        page_size: Resultados por página (max 50)
        delay: Delay entre requisições (segundos)
        max_pages: Limite de páginas (None = todas)
        progress_callback: Callable(page: int, total_so_far: int)

    Returns:
        Lista de dicts com: Data, Sessao, Orador, Partido, Estado, Fase, Hora, Discurso
    """
    _cb = progress_callback or (lambda *a: None)
    results = []
    try:
        for page, _total, rows in iter_listing(
            keywords, data_inicio, data_fim, page_size, max_pages,
        ):
            for row in rows:
                url = row.pop("url")
                text = fetch_speech(url)
                row["Discurso"] = text if text is not None else "[Erro: falha ao baixar discurso]"
                results.append(row)
                time.sleep(delay)
            _cb(page, len(results))
    except ListingError as e:
        print(e)
    return results


def iter_listing(
    keywords: list[str],
    data_inicio: str,
    data_fim: str,
    page_size: int = 50,
    max_pages: int | None = None,
) -> Iterator[tuple[int, int | None, list[dict]]]:
    """
    Percorre as páginas de resultados do Plenário.

    Yields:
        (página, total de discursos encontrados, linhas da página). Cada linha tem
        Data, Sessao, Orador, Partido, Estado, Fase, Hora e url (link do texto).

    Raises:
        ListingError: se uma página falhar mesmo após as novas tentativas.
    """
    params = {
        "CurrentPage": 1,
        "txOrador": "",
        "txPartido": "",
        "txUF": "",
        "dtInicio": data_inicio,
        "dtFim": data_fim,
        "txTexto": _search_expression(keywords),
        "txSumario": "",
        # committee results ("comissao") have no speaker/party columns, which the analysis needs
        "basePesq": "plenario",
        "CampoOrdenacao": "dtSessao",
        "TipoOrdenacao": "DESC",
        "PageSize": page_size,
    }

    page = 1
    while not (max_pages and page > max_pages):
        params["CurrentPage"] = page
        resp = _get(BASE_URL, params=params, timeout=60)
        if resp is None:
            raise ListingError(f"Página {page} da listagem falhou após {len(RETRY_WAITS) + 1} tentativas")

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = []
        for tr in soup.select("table tbody tr"):
            cols = tr.find_all("td")
            if len(cols) >= 6 and (row := _parse_row(cols)):
                rows.append(row)
        if not rows:
            return

        yield page, _parse_total(soup), rows

        if not soup.find("a", string=lambda t: t and "Próxima" in t):
            return
        page += 1


def count_speeches(data_inicio: str, data_fim: str) -> int | None:
    """
    Total de discursos do Plenário no período (todos os temas), lido do contador da busca.

    Args:
        data_inicio, data_fim: DD/MM/YYYY

    Returns:
        O total, 0 se não houver discursos, ou None se o servidor não responder.
    """
    params = {
        "CurrentPage": 1, "txOrador": "", "txPartido": "", "txUF": "",
        "dtInicio": data_inicio, "dtFim": data_fim, "txTexto": "", "txSumario": "",
        "basePesq": "plenario", "CampoOrdenacao": "dtSessao", "TipoOrdenacao": "DESC",
        "PageSize": 1,  # only the counter is needed
    }
    resp = _get(BASE_URL, params=params, timeout=60)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    total = _parse_total(soup)
    if total is not None:
        return total
    return 0 if "Nenhum discurso encontrado" in soup.get_text(" ") else None


def fetch_speech(url: str) -> str | None:
    """Baixa o texto de um discurso. None se o servidor não responder."""
    resp = _get(url, timeout=30)
    return _extract_speech_text(resp.text) if resp is not None else None


def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response | None:
    """GET com novas tentativas e espera crescente para erros temporários do servidor."""
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        t0 = time.time()
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                # pages are UTF-8 but the server declares latin-1
                resp.encoding = "utf-8"
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


def _search_expression(keywords: list[str]) -> str:
    """
    Cada keyword com e sem acento, unidas por OR.

    O índice do SitaQWeb mudou no fim de 2022: discursos anteriores só são encontrados pela
    forma acentuada ("feminicídio") e os posteriores só pela forma sem acento ("feminicidio").
    """
    terms = []
    for kw in keywords:
        for form in (kw, _strip_accents(kw)):
            if form not in terms:
                terms.append(form)
    return " OR ".join(f'"{t}"' for t in terms)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _parse_total(soup: BeautifulSoup) -> int | None:
    """Lê o total em '1 a 50 de 148 documentos encontrados'."""
    match = re.search(r"de\s+([\d.]+)\s+documentos encontrados", soup.get_text(" "))
    return int(match.group(1).replace(".", "")) if match else None


def _parse_row(cols) -> dict | None:
    """Extrai metadados e o link do texto de uma linha da tabela de resultados."""
    link_elem = cols[3].find("a")
    if not link_elem or "href" not in link_elem.attrs:
        return None

    # the href is split across lines with tabs; spaces inside values (names) must stay
    href = re.sub(r"[\r\n\t]", "", link_elem["href"])
    nome, partido, estado = _parse_orador(cols[5].get_text(strip=True))

    return {
        "Data": cols[0].get_text(strip=True),
        "Sessao": cols[1].get_text(strip=True),
        "Orador": nome,
        "Partido": partido,
        "Estado": estado,
        "Fase": cols[2].get_text(strip=True),
        "Hora": cols[6].get_text(strip=True) if len(cols) > 6 else "",
        "url": urljoin(DETAIL_BASE, href.strip()),
    }


# deputies come as "NOME, PARTIDO-UF" ("CABO DACIOLO, S.PART.-RJ")
_DEPUTADO = re.compile(r"^(?P<nome>.+?)\s*,\s*(?P<partido>[^,]+?)\s*-\s*(?P<uf>[A-Z]{2})$")
CONVIDADO = "CONVIDADO"


def _parse_orador(raw: str) -> tuple[str, str, str]:
    """
    Extrai nome, partido e estado do campo orador.

    Quem não está no formato de deputado é convidado (sessões solenes, comissões gerais):
    "DILMA ROUSSEFF, MINISTRA-CHEFE DA CASA CIVIL", "MÁRCIA ROLLEMBERG (PRIMEIRA-DAMA ...)".
    O texto inteiro fica como nome, porque o cargo identifica a pessoa, e o partido é CONVIDADO.
    """
    raw = " ".join(raw.split())
    m = _DEPUTADO.match(raw)
    if m:
        return m["nome"], m["partido"], m["uf"]
    return raw or "SEM NOME", CONVIDADO, ""


def _extract_speech_text(html_str: str) -> str:
    """Extrai texto do discurso da página de detalhe. Preserva acentos."""
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
