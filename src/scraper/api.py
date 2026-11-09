"""
Scraper via API Dados Abertos da Câmara dos Deputados.

Itera por deputados, puxa discursos, filtra por keywords em Python.
Texto original preservado (campo `transcricao`).
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

API_BASE = "https://dadosabertos.camara.leg.br/api/v2"

HEADERS = {"Accept": "application/json"}


def scrape(
    keywords: list[str],
    data_inicio: str = "2000-01-01",
    data_fim: str = "2025-12-31",
    max_workers: int = 5,
    delay: float = 0.3,
    max_deputados: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """
    Busca discursos via API Dados Abertos.

    Fluxo: lista deputados → puxa discursos de cada um → filtra por keywords.

    Args:
        keywords: Termos de busca (filtrados no texto da transcrição)
        data_inicio: YYYY-MM-DD
        data_fim: YYYY-MM-DD
        max_workers: Requisições paralelas
        delay: Delay entre requisições por worker
        max_deputados: Limita nº de deputados (pra testes)
        progress_callback: Callable(deputados_done: int, total_deputados: int, speeches_found: int)

    Returns:
        Lista de dicts com: Data, Sessao, Orador, Partido, Estado, Fase, Hora, Discurso
    """
    _cb = progress_callback or (lambda *a: None)

    kw_pattern = re.compile(
        "|".join(re.escape(kw) for kw in keywords),
        re.IGNORECASE,
    )

    print("Listando deputados...")
    deputados = _list_deputados()
    if max_deputados:
        deputados = deputados[:max_deputados]
    print(f"Total de deputados: {len(deputados)}")

    results = []
    done = 0

    def process_deputado(dep):
        dep_speeches = _get_speeches_for_deputado(
            dep["id"], data_inicio, data_fim, delay,
        )
        matched = []
        for speech in dep_speeches:
            text = speech.get("transcricao", "") or ""
            if kw_pattern.search(text):
                matched.append(_normalize_record(speech, dep))
        return matched

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_deputado, dep): dep
            for dep in deputados
        }
        for future in as_completed(futures):
            try:
                matched = future.result()
                results.extend(matched)
            except Exception as e:
                dep = futures[future]
                print(f"Erro no deputado {dep.get('nome', '?')}: {e}")
            done += 1
            _cb(done, len(deputados), len(results))

    return results


def _list_deputados() -> list[dict]:
    """Lista todos os deputados (todas as legislaturas)."""
    deputados = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/deputados",
            params={"itens": 100, "pagina": page, "ordem": "ASC", "ordenarPor": "nome"},
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code != 200:
            break
        data = resp.json().get("dados", [])
        if not data:
            break
        deputados.extend(data)
        page += 1
    return deputados


def _get_speeches_for_deputado(
    dep_id: int,
    data_inicio: str,
    data_fim: str,
    delay: float,
) -> list[dict]:
    """Puxa todos os discursos de um deputado no range de datas."""
    speeches = []
    page = 1
    while True:
        try:
            resp = requests.get(
                f"{API_BASE}/deputados/{dep_id}/discursos",
                params={
                    "dataInicio": data_inicio,
                    "dataFim": data_fim,
                    "itens": 100,
                    "pagina": page,
                    "ordenarPor": "dataHoraInicio",
                    "ordem": "ASC",
                },
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(30)
                continue
            if resp.status_code != 200:
                break
            data = resp.json().get("dados", [])
            if not data:
                break
            speeches.extend(data)
            page += 1
            time.sleep(delay)
        except requests.exceptions.RequestException:
            break
    return speeches


def _normalize_record(speech: dict, deputado: dict) -> dict:
    """Converte registro da API pro formato padrão do pipeline."""
    dt_str = speech.get("dataHoraInicio", "")
    try:
        dt = datetime.fromisoformat(dt_str)
        data = dt.strftime("%d/%m/%Y")
        hora = dt.strftime("%Hh%M")
    except (ValueError, TypeError):
        data = ""
        hora = ""

    fase_obj = speech.get("faseEvento") or {}
    fase = fase_obj.get("titulo", speech.get("tipoDiscurso", ""))

    return {
        "Data": data,
        "Sessao": "",
        "Orador": deputado.get("nome", ""),
        "Partido": deputado.get("siglaPartido", ""),
        "Estado": deputado.get("siglaUf", ""),
        "Fase": fase,
        "Hora": hora,
        "Discurso": speech.get("transcricao", ""),
    }
