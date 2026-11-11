"""
API do app (FastAPI).

Rodar:  python -m src.app   → http://127.0.0.1:8000/docs para testar os endpoints
"""

import asyncio
import io
import json
import re
import shutil
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from ..pipeline.model_config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, TopicModelConfig, methodology
from ..scraper.accents import accented_forms
from . import db, jobs

FINAL_STATUSES = {"done", "interrupted"}
RUN_FINAL_STATUSES = {"done", "failed"}
STATIC = Path(__file__).parent / "static"
ACCENTS = re.compile(r"[áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ]")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    conn = db.connect()
    db.init_db(conn)
    conn.close()
    yield


app = FastAPI(title="Discursos da Câmara", lifespan=lifespan)


class ScrapeRequest(BaseModel):
    keywords: list[str] = Field(min_length=1, examples=[["feminicídio"]])
    data_inicio: date = Field(examples=["2023-01-01"])
    data_fim: date = Field(examples=["2023-12-31"])
    name: str | None = None

    @field_validator("keywords")
    @classmethod
    def strip_keywords(cls, v: list[str]) -> list[str]:
        v = [k.strip() for k in v if k.strip()]
        if not v:
            raise ValueError("informe ao menos uma keyword")
        return v

    @model_validator(mode="after")
    def check_dates(self):
        if self.data_inicio > self.data_fim:
            raise ValueError("data_inicio deve ser anterior a data_fim")
        return self


@app.get("/api/datasets")
def list_datasets():
    conn = db.connect()
    try:
        return [_with_running(d) for d in db.list_datasets(conn)]
    finally:
        conn.close()


@app.post("/api/datasets/scrape", status_code=201)
def create_scrape(req: ScrapeRequest):
    name = req.name or f"{', '.join(req.keywords)} ({req.data_inicio:%Y}–{req.data_fim:%Y})"
    conn = db.connect()
    try:
        dataset_id = db.create_scrape_dataset(
            conn, name, req.keywords, req.data_inicio.isoformat(), req.data_fim.isoformat(),
        )
        jobs.start_scrape(dataset_id)
        return _with_running(db.get_dataset(conn, dataset_id))
    finally:
        conn.close()


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    conn = db.connect()
    try:
        return _with_running(_require(conn, dataset_id))
    finally:
        conn.close()


@app.post("/api/datasets/{dataset_id}/resume")
def resume_scrape(dataset_id: int):
    conn = db.connect()
    try:
        ds = _require(conn, dataset_id)
        if ds["source"] != "sitaqweb":
            raise HTTPException(400, "Só coletas do SitaQWeb podem ser retomadas")
        if ds["status"] == "done":
            raise HTTPException(409, "Coleta já concluída")
        if not jobs.start_scrape(dataset_id):
            raise HTTPException(409, "Coleta já está rodando")
        return _with_running(db.get_dataset(conn, dataset_id))
    finally:
        conn.close()


@app.delete("/api/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int):
    conn = db.connect()
    try:
        _require(conn, dataset_id)
        if jobs.is_running(dataset_id):
            raise HTTPException(409, "A coleta ainda está rodando; aguarde ela terminar para excluir")
        if any(r["dataset_id"] == dataset_id and r["status"] in ("pending", "running") for r in db.list_runs(conn)):
            raise HTTPException(409, "Há uma análise deste dataset rodando; aguarde ela terminar para excluir")
        for run_id in db.delete_dataset(conn, dataset_id):
            shutil.rmtree(db.run_dir(run_id), ignore_errors=True)
    finally:
        conn.close()


@app.get("/api/datasets/{dataset_id}/events")
async def dataset_events(dataset_id: int, request: Request):
    """Progresso da coleta via Server-Sent Events. Termina quando a coleta para."""
    conn = db.connect()
    try:
        _require(conn, dataset_id)
    finally:
        conn.close()

    async def stream():
        last = None
        conn = db.connect()
        try:
            while not await request.is_disconnected():
                ds = _with_running(db.get_dataset(conn, dataset_id))
                payload = json.dumps(ds, ensure_ascii=False)
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if ds["status"] in FINAL_STATUSES and not ds["running"]:
                    return
                await asyncio.sleep(1)
        finally:
            conn.close()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/datasets/{dataset_id}/csv")
def export_csv(dataset_id: int):
    conn = db.connect()
    try:
        ds = _require(conn, dataset_id)
        df = db.dataset_dataframe(conn, dataset_id)
    finally:
        conn.close()
    # utf-8-sig: the BOM makes Excel read accents correctly
    content = df.to_csv(index=False).encode("utf-8-sig")
    return Response(
        content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="dataset_{ds["id"]}.csv"'},
    )


@app.post("/api/datasets/import", status_code=201)
async def import_csv(file: UploadFile = File(...), name: str | None = Form(None)):
    """'Já possuo CSV': importa um CSV no formato novo (texto original, com acentos)."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        raise HTTPException(400, "O arquivo não está em UTF-8. Salve o CSV como UTF-8 e tente de novo.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise HTTPException(400, f"Não foi possível ler o CSV: {e}")

    df = df.rename(columns={"Sessão": "Sessao"})
    missing = [c for c in db.CSV_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Colunas faltando: {', '.join(missing)}. Esperado: {', '.join(db.CSV_COLUMNS)}.")

    df = df[db.CSV_COLUMNS].fillna("")
    df = df[df["Discurso"].str.strip() != ""]
    if df.empty:
        raise HTTPException(400, "Nenhum discurso com texto no arquivo.")

    long_texts = df["Discurso"][df["Discurso"].str.len() > 200]
    if len(long_texts) and long_texts.str.contains(ACCENTS).mean() < 0.5:
        raise HTTPException(
            400,
            "Os textos parecem estar sem acentos (formato antigo, normalizado). O app só aceita texto original, "
            "porque os embeddings perdem qualidade sem acentos. Colete de novo pelo app (\"Criar do zero\").",
        )

    conn = db.connect()
    try:
        dataset_id = db.create_csv_dataset(conn, name or Path(file.filename or "importado.csv").stem, df)
        return db.get_dataset(conn, dataset_id)
    finally:
        conn.close()


class RunRequest(BaseModel):
    dataset_id: int
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    nr_topics: int | None = Field(default=None, ge=2, le=200)  # None = sem redução (todos os tópicos do HDBSCAN)
    threshold: float = Field(default=TopicModelConfig.multi_label_threshold, ge=0.0, le=1.0)

    @field_validator("embedding_model")
    @classmethod
    def known_model(cls, v: str) -> str:
        if v not in EMBEDDING_MODELS:
            raise ValueError(f"modelo desconhecido; opções: {', '.join(EMBEDDING_MODELS)}")
        return v


@app.get("/api/options")
def options():
    """Valores padrão e opções dos controles da análise."""
    return {
        "embedding_models": [
            {"id": k, "label": s.label, "descricao": s.descricao, "min_cluster_size": s.min_cluster_size}
            for k, s in EMBEDDING_MODELS.items()
        ],
        "defaults": RunRequest.model_construct(dataset_id=0).model_dump(exclude={"dataset_id"}),
        "min_docs": jobs.MIN_DOCS,
    }


@app.get("/api/accents")
def accents(words: list[str] = Query(default=[], max_length=200)):
    """
    Formas acentuadas das palavras digitadas sem acento nas keywords.

    Só volta as palavras que têm forma acentuada: {"feminicidio": ["feminicídio"], "pais": ["país", "pais"]}.
    """
    return {w: forms for w in dict.fromkeys(words) if (forms := accented_forms(w))}


@app.get("/api/methodology")
def methodology_table(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    nr_topics: int = 20,
    sem_reducao: bool = False,
    threshold: float = TopicModelConfig.multi_label_threshold,
):
    """Tabela de parâmetros para os valores escolhidos (painel de documentação)."""
    if embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(400, f"modelo desconhecido; opções: {', '.join(EMBEDDING_MODELS)}")
    cfg = TopicModelConfig(
        embedding_model=embedding_model,
        nr_topics=None if sem_reducao else nr_topics,
        multi_label_threshold=threshold,
    )
    return methodology(cfg)


@app.get("/api/estimate")
def estimate(dataset_id: int, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """Tempo estimado da análise nesta máquina (GPU ou CPU), descontando embeddings já guardados."""
    if embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(400, f"modelo desconhecido; opções: {', '.join(EMBEDDING_MODELS)}")
    conn = db.connect()
    try:
        _require(conn, dataset_id)
        return jobs.estimate(conn, dataset_id, embedding_model)
    finally:
        conn.close()


@app.get("/api/runs")
def list_runs():
    conn = db.connect()
    try:
        return db.list_runs(conn)
    finally:
        conn.close()


@app.post("/api/runs", status_code=201)
def create_run(req: RunRequest):
    conn = db.connect()
    try:
        _require(conn, req.dataset_id)
        if jobs.is_running(req.dataset_id):
            raise HTTPException(409, "A coleta deste dataset ainda está rodando")
        if jobs.pipeline_busy():
            raise HTTPException(409, "Já existe uma análise rodando; aguarde ela terminar")
        run_id = db.create_run(conn, req.dataset_id, req.model_dump(exclude={"dataset_id"}))
        if not jobs.start_pipeline(run_id):
            db.update_run(conn, run_id, status="failed", error="Outra análise começou ao mesmo tempo")
            raise HTTPException(409, "Já existe uma análise rodando; aguarde ela terminar")
        return db.get_run(conn, run_id)
    finally:
        conn.close()


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    conn = db.connect()
    try:
        return _require_run(conn, run_id)
    finally:
        conn.close()


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: int):
    conn = db.connect()
    try:
        run = _require_run(conn, run_id)
        if run["status"] in ("pending", "running"):
            raise HTTPException(409, "A análise ainda está rodando; aguarde ela terminar para excluir")
        db.delete_run(conn, run_id)
        shutil.rmtree(db.run_dir(run_id), ignore_errors=True)
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: int, request: Request):
    """Progresso da análise via Server-Sent Events."""
    conn = db.connect()
    try:
        _require_run(conn, run_id)
    finally:
        conn.close()

    async def stream():
        last = None
        conn = db.connect()
        try:
            while not await request.is_disconnected():
                run = db.get_run(conn, run_id)
                payload = json.dumps(run, ensure_ascii=False)
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if run["status"] in RUN_FINAL_STATUSES:
                    return
                await asyncio.sleep(1)
        finally:
            conn.close()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/dashboard_data.json")
def run_dashboard_data(run_id: int):
    return FileResponse(_run_file(run_id, "dashboard_data.json"), media_type="application/json",
                        filename=f"dashboard_data_{run_id}.json")


@app.get("/api/runs/{run_id}/discursos_processados.csv")
def run_processed_csv(run_id: int):
    return FileResponse(_run_file(run_id, "discursos_processados.csv"), media_type="text/csv",
                        filename=f"discursos_processados_{run_id}.csv")


@app.get("/api/runs/{run_id}/dashboard.html")
def run_dashboard_standalone(run_id: int):
    """Dashboard com os dados embutidos: abre em qualquer navegador, sem o app."""
    # "</" inside the JSON would end the <script> element early
    data = _run_file(run_id, "dashboard_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    inline = f"<script>window.DASHBOARD_DATA = {data};</script>"
    page = (STATIC / "dashboard.html").read_text(encoding="utf-8").replace("<!--DASHBOARD_DATA-->", inline, 1)
    return HTMLResponse(page, headers={"Content-Disposition": f'attachment; filename="dashboard_{run_id}.html"'})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "dashboard.html")


def _run_file(run_id: int, name: str) -> Path:
    conn = db.connect()
    try:
        run = _require_run(conn, run_id)
    finally:
        conn.close()
    path = db.run_dir(run_id) / name
    if run["status"] != "done" or not path.exists():
        raise HTTPException(404, "Resultado ainda não disponível")
    return path


def _require_run(conn, run_id: int) -> dict:
    run = db.get_run(conn, run_id)
    if run is None:
        raise HTTPException(404, "Análise não encontrada")
    return run


def _require(conn, dataset_id: int) -> dict:
    ds = db.get_dataset(conn, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset não encontrado")
    return ds


def _with_running(ds: dict) -> dict:
    return {**ds, "running": jobs.is_running(ds["id"])}
