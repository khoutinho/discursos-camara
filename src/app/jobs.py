"""
Tarefas em segundo plano (threads), com o progresso gravado no banco.

Retomar = rodar de novo: a listagem é percorrida outra vez (rápido) e só os
discursos ainda sem texto são baixados (lento).
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from ..scraper import sitaqweb
from . import db

FETCH_WORKERS = 3

_running: set[int] = set()
_lock = threading.Lock()


def start_scrape(dataset_id: int) -> bool:
    """Inicia a coleta em segundo plano. False se ela já estiver rodando."""
    with _lock:
        if dataset_id in _running:
            return False
        _running.add(dataset_id)
    threading.Thread(target=_run_scrape, args=(dataset_id,), daemon=True).start()
    return True


def is_running(dataset_id: int) -> bool:
    return dataset_id in _running


def _run_scrape(dataset_id: int) -> None:
    conn = db.connect()
    try:
        ds = db.get_dataset(conn, dataset_id)
        db.set_status(conn, dataset_id, "running")

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            for _page, total, rows in sitaqweb.iter_listing(
                ds["keywords"], _to_br_date(ds["data_inicio"]), _to_br_date(ds["data_fim"]),
            ):
                db.set_total(conn, dataset_id, total)
                _fetch(conn, pool, db.add_listing_rows(conn, dataset_id, rows))

            # second chance for speeches that failed earlier in this run or in a previous one
            _fetch(conn, pool, db.pending_speeches(conn, dataset_id))

        failed = len(db.pending_speeches(conn, dataset_id))
        if failed:
            db.set_status(conn, dataset_id, "interrupted",
                          f"{failed} discursos não foram baixados; retome a coleta para tentar de novo")
        else:
            db.set_status(conn, dataset_id, "done")
    except sitaqweb.ListingError as e:
        db.set_status(conn, dataset_id, "interrupted", f"{e}. Retome a coleta para continuar.")
    except Exception as e:
        db.set_status(conn, dataset_id, "interrupted", f"Erro inesperado: {e!r}")
        raise
    finally:
        conn.close()
        with _lock:
            _running.discard(dataset_id)


def _fetch(conn, pool: ThreadPoolExecutor, pending: list[tuple[int, str]]) -> None:
    ids = [speech_id for speech_id, _ in pending]
    texts = pool.map(sitaqweb.fetch_speech, [url for _, url in pending])
    for speech_id, text in zip(ids, texts):
        db.save_speech_text(conn, speech_id, text)


MIN_DOCS = 50

# overall progress (0–1) at the start of each stage; totals fill 0.02→0.1, embeddings 0.1→0.75
STAGE_PROGRESS = {
    "loading": 0.0, "totals": 0.02, "embeddings": 0.1, "clustering": 0.75, "reduce": 0.88, "statistics": 0.93,
}

# transcripts are published with a delay; a month's total only counts as final this long after it ends
TOTALS_SETTLE_DAYS = 90

_pipeline_lock = threading.Lock()


def start_pipeline(run_id: int) -> bool:
    """Inicia a análise em segundo plano. False se outra análise já estiver rodando."""
    if not _pipeline_lock.acquire(blocking=False):
        return False
    threading.Thread(target=_run_pipeline, args=(run_id,), daemon=True).start()
    return True


def pipeline_busy() -> bool:
    return _pipeline_lock.locked()


def _run_pipeline(run_id: int) -> None:
    conn = db.connect()
    try:
        # heavy imports (torch, BERTopic) only when an analysis actually runs
        from ..pipeline.export import export_dashboard_json, export_processed_csv
        from ..pipeline.preprocessing import clean
        from ..pipeline.run import analyze
        from ..pipeline.topic_modeling import TopicModelConfig

        run = db.get_run(conn, run_id)
        params = run["params"]
        stage = _stage_reporter(conn, run_id)
        stage("loading")
        db.update_run(conn, run_id, status="running")

        df = clean(db.dataset_dataframe(conn, run["dataset_id"], with_id=True)).reset_index(drop=True)
        if len(df) < MIN_DOCS:
            raise PipelineError(
                f"O dataset tem {len(df)} discursos com texto e data válida; são necessários ao menos {MIN_DOCS} "
                "para encontrar tópicos. Amplie o período ou as keywords."
            )

        ds = db.get_dataset(conn, run["dataset_id"])
        if ds["data_inicio"]:
            period = (date.fromisoformat(ds["data_inicio"]), date.fromisoformat(ds["data_fim"]))
        else:  # imported CSV: the period is what the file covers
            period = (df["Data_dt"].min().date(), df["Data_dt"].max().date())
        plenary_totals = _plenary_totals(conn, run_id, *period)

        embeddings = _embeddings_for(conn, run_id, df, params["embedding_model"])

        config = TopicModelConfig(
            embedding_model=params["embedding_model"],
            nr_topics=params["nr_topics"],
            multi_label_threshold=params["threshold"],
        )
        df, stats = analyze(
            df, config, embeddings=embeddings, progress_callback=lambda s, _p: stage(s),
            period=period, plenary_totals=plenary_totals,
        )

        stats["info"] = {
            "run_id": run_id,
            "dataset": ds["name"],
            "fonte": ds["source"],
            "keywords": ds["keywords"],
            "periodo": [ds["data_inicio"], ds["data_fim"]] if ds["data_inicio"] else None,
            "parametros": params,
            "gerado_em": run["created_at"],
        }

        out = db.run_dir(run_id)
        export_dashboard_json(stats, out / "dashboard_data.json")
        export_processed_csv(df, out / "discursos_processados.csv")
        db.update_run(conn, run_id, status="done", stage="done", progress=1.0, summary=stats["modelo"])
    except PipelineError as e:
        db.update_run(conn, run_id, status="failed", error=str(e))
    except Exception as e:
        db.update_run(conn, run_id, status="failed", error=f"Erro inesperado: {e!r}")
        raise
    finally:
        conn.close()
        _pipeline_lock.release()


class PipelineError(Exception):
    """Erro esperado, com mensagem para mostrar na tela."""


def _stage_reporter(conn, run_id: int):
    def report(stage: str, progress: float | None = None) -> None:
        db.update_run(conn, run_id, stage=stage,
                      progress=progress if progress is not None else STAGE_PROGRESS.get(stage))
    return report


def _plenary_totals(conn, run_id: int, inicio: date, fim: date) -> dict[str, int | None]:
    """
    Total de discursos do Plenário em cada mês do período, para as tendências em proporção.

    Meses inteiros vêm do cache quando já estão consolidados; meses recentes são consultados de
    novo. Meses pela metade (pontas do período) são consultados no trecho exato e não vão pro cache.
    Um mês que falhar fica None e é buscado de novo na próxima análise.
    """
    today = date.today()
    cache = db.cached_month_totals(conn)
    totals: dict[str, int | None] = {}
    to_fetch: list[tuple[str, date, date, bool]] = []  # (month, start, end, cacheable)

    month_start = inicio.replace(day=1)
    while month_start <= fim:
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        key = month_start.strftime("%Y-%m")
        start, end = max(month_start, inicio), min(month_end, fim)
        whole = start == month_start and end == month_end
        cached = cache.get(key)
        if whole and cached and date.fromisoformat(cached[1]) >= month_end + timedelta(days=TOTALS_SETTLE_DAYS):
            totals[key] = cached[0]
        else:
            to_fetch.append((key, start, end, whole))
        month_start = month_end + timedelta(days=1)

    start_p, end_p = STAGE_PROGRESS["totals"], STAGE_PROGRESS["embeddings"]
    db.update_run(conn, run_id, stage="totals", progress=start_p)
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        results = pool.map(
            lambda m: sitaqweb.count_speeches(m[1].strftime("%d/%m/%Y"), m[2].strftime("%d/%m/%Y")), to_fetch,
        )
        for i, ((key, _start, _end, whole), total) in enumerate(zip(to_fetch, results), start=1):
            totals[key] = total
            if total is not None and whole:
                db.save_month_total(conn, key, total)
            if i % 12 == 0 or i == len(to_fetch):
                db.update_run(conn, run_id, progress=start_p + (end_p - start_p) * i / len(to_fetch))
    return totals


def _embeddings_for(conn, run_id: int, df, model: str):
    """Embeddings na ordem de df: usa o cache do banco e calcula só os que faltam."""
    import numpy as np
    from ..pipeline.topic_modeling import iter_encode

    ids = df["speech_id"].tolist()
    cached = db.load_embeddings(conn, ids, model)
    missing = [i for i, sid in enumerate(ids) if sid not in cached]

    start, end = STAGE_PROGRESS["embeddings"], STAGE_PROGRESS["clustering"]
    db.update_run(conn, run_id, stage="embeddings", progress=start + (end - start) * (1 - len(missing) / len(ids)))
    if missing:
        texts = df["Discurso"].iloc[missing].tolist()
        for offset, vectors in iter_encode(texts, model):
            chunk_ids = [ids[i] for i in missing[offset:offset + len(vectors)]]
            db.save_embeddings(conn, chunk_ids, model, vectors)
            cached.update(zip(chunk_ids, vectors))
            done = len(ids) - len(missing) + offset + len(vectors)
            db.update_run(conn, run_id, progress=start + (end - start) * done / len(ids))
    return np.vstack([cached[sid] for sid in ids])


def _to_br_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"
