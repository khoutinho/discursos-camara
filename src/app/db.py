"""
Banco SQLite do app.

- datasets: cada coleta (ou importação), com parâmetros e status
- speeches: discursos únicos, identificados pelo link no SitaQWeb; discurso NULL = ainda não baixado
- dataset_speeches: quais discursos pertencem a cada dataset

Cada thread deve abrir a própria conexão com connect().
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/camara.db")

CSV_COLUMNS = ["Data", "Sessao", "Orador", "Partido", "Estado", "Fase", "Hora", "Discurso"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    keywords TEXT,
    data_inicio TEXT,
    data_fim TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    total_expected INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speeches (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    data TEXT, sessao TEXT, orador TEXT, partido TEXT, estado TEXT, fase TEXT, hora TEXT,
    discurso TEXT,
    fetch_failures INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dataset_speeches (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    speech_id INTEGER NOT NULL REFERENCES speeches(id),
    PRIMARY KEY (dataset_id, speech_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    speech_id INTEGER NOT NULL REFERENCES speeches(id),
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (speech_id, model)
);

-- segundos por discurso medidos nesta máquina, por modelo e dispositivo (estimativa de tempo)
CREATE TABLE IF NOT EXISTS embedding_speed (
    model TEXT NOT NULL,
    device TEXT NOT NULL,
    sec_per_doc REAL NOT NULL,
    n_docs INTEGER NOT NULL,
    measured_at TEXT NOT NULL,
    PRIMARY KEY (model, device)
);

CREATE TABLE IF NOT EXISTS plenary_month_totals (
    month TEXT PRIMARY KEY,
    total INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    params TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT,
    progress REAL,
    error TEXT,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

RUNS_DIR = Path("data/runs")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # a 'running' dataset at startup means the app was closed mid-scrape
    conn.execute(
        "UPDATE datasets SET status='interrupted', error='App encerrado durante a coleta', updated_at=? "
        "WHERE status='running'",
        (_now(),),
    )
    conn.execute(
        "UPDATE runs SET status='failed', error='App encerrado durante a análise', updated_at=? "
        "WHERE status IN ('pending', 'running')",
        (_now(),),
    )
    conn.commit()


def create_scrape_dataset(
    conn: sqlite3.Connection,
    name: str,
    keywords: list[str],
    data_inicio: str,
    data_fim: str,
) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO datasets (name, source, keywords, data_inicio, data_fim, created_at, updated_at) "
        "VALUES (?, 'sitaqweb', ?, ?, ?, ?, ?)",
        (name, json.dumps(keywords, ensure_ascii=False), data_inicio, data_fim, now, now),
    )
    conn.commit()
    return cur.lastrowid


def create_csv_dataset(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> int:
    """Cria um dataset a partir de um CSV já validado (colunas de CSV_COLUMNS)."""
    now = _now()
    cur = conn.execute(
        "INSERT INTO datasets (name, source, status, total_expected, created_at, updated_at) "
        "VALUES (?, 'csv', 'done', ?, ?, ?)",
        (name, len(df), now, now),
    )
    dataset_id = cur.lastrowid
    for row in df[CSV_COLUMNS].itertuples(index=False):
        speech = conn.execute(
            "INSERT INTO speeches (data, sessao, orador, partido, estado, fase, hora, discurso) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(row),
        )
        conn.execute(
            "INSERT INTO dataset_speeches (dataset_id, speech_id) VALUES (?, ?)",
            (dataset_id, speech.lastrowid),
        )
    conn.commit()
    return dataset_id


def get_dataset(conn: sqlite3.Connection, dataset_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT d.*,
               COUNT(s.id) AS listed,
               COUNT(s.discurso) AS fetched,
               SUM(CASE WHEN s.discurso IS NULL AND s.fetch_failures > 0 THEN 1 ELSE 0 END) AS failed
        FROM datasets d
        LEFT JOIN dataset_speeches ds ON ds.dataset_id = d.id
        LEFT JOIN speeches s ON s.id = ds.speech_id
        WHERE d.id = ?
        GROUP BY d.id
        """,
        (dataset_id,),
    ).fetchone()
    return _dataset_dict(row) if row else None


def list_datasets(conn: sqlite3.Connection) -> list[dict]:
    ids = [r["id"] for r in conn.execute("SELECT id FROM datasets ORDER BY id DESC")]
    return [get_dataset(conn, i) for i in ids]


def set_status(conn: sqlite3.Connection, dataset_id: int, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE datasets SET status=?, error=?, updated_at=? WHERE id=?",
        (status, error, _now(), dataset_id),
    )
    conn.commit()


def set_total(conn: sqlite3.Connection, dataset_id: int, total: int | None) -> None:
    if total is not None:
        conn.execute("UPDATE datasets SET total_expected=? WHERE id=?", (total, dataset_id))
        conn.commit()


def add_listing_rows(conn: sqlite3.Connection, dataset_id: int, rows: list[dict]) -> list[tuple[int, str]]:
    """Registra as linhas de uma página da listagem. Retorna (id, url) das que ainda não têm texto."""
    pending = []
    for r in rows:
        conn.execute(
            "INSERT INTO speeches (url, data, sessao, orador, partido, estado, fase, hora) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(url) DO NOTHING",
            (r["url"], r["Data"], r["Sessao"], r["Orador"], r["Partido"], r["Estado"], r["Fase"], r["Hora"]),
        )
        speech = conn.execute("SELECT id, discurso FROM speeches WHERE url=?", (r["url"],)).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO dataset_speeches (dataset_id, speech_id) VALUES (?, ?)",
            (dataset_id, speech["id"]),
        )
        if speech["discurso"] is None:
            pending.append((speech["id"], r["url"]))
    conn.commit()
    return pending


def pending_speeches(conn: sqlite3.Connection, dataset_id: int) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT s.id, s.url FROM speeches s JOIN dataset_speeches ds ON ds.speech_id = s.id "
        "WHERE ds.dataset_id = ? AND s.discurso IS NULL",
        (dataset_id,),
    )
    return [(r["id"], r["url"]) for r in rows]


def save_speech_text(conn: sqlite3.Connection, speech_id: int, text: str | None) -> None:
    if text is None:
        conn.execute("UPDATE speeches SET fetch_failures = fetch_failures + 1 WHERE id=?", (speech_id,))
    else:
        conn.execute("UPDATE speeches SET discurso=? WHERE id=?", (text, speech_id))
    conn.commit()


def dataset_dataframe(conn: sqlite3.Connection, dataset_id: int, with_id: bool = False) -> pd.DataFrame:
    """Discursos baixados do dataset, no formato de CSV que o pipeline aceita."""
    df = pd.read_sql_query(
        """
        SELECT s.id AS speech_id, s.data AS Data, s.sessao AS Sessao, s.orador AS Orador,
               s.partido AS Partido, s.estado AS Estado, s.fase AS Fase, s.hora AS Hora,
               s.discurso AS Discurso
        FROM speeches s JOIN dataset_speeches ds ON ds.speech_id = s.id
        WHERE ds.dataset_id = ? AND s.discurso IS NOT NULL AND s.discurso != ''
        ORDER BY s.id
        """,
        conn,
        params=(dataset_id,),
    )
    return df if with_id else df[CSV_COLUMNS]


def load_embeddings(conn: sqlite3.Connection, speech_ids: list[int], model: str) -> dict[int, np.ndarray]:
    found = {}
    for start in range(0, len(speech_ids), 900):  # SQLite caps bound parameters per query
        chunk = speech_ids[start:start + 900]
        rows = conn.execute(
            f"SELECT speech_id, vector FROM embeddings WHERE model=? AND speech_id IN ({','.join('?' * len(chunk))})",
            (model, *chunk),
        )
        for r in rows:
            found[r["speech_id"]] = np.frombuffer(r["vector"], dtype=np.float32)
    return found


def save_embeddings(conn: sqlite3.Connection, speech_ids: list[int], model: str, vectors: np.ndarray) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (speech_id, model, vector) VALUES (?, ?, ?)",
        [(sid, model, vec.astype(np.float32).tobytes()) for sid, vec in zip(speech_ids, vectors)],
    )
    conn.commit()


def count_embeddings(conn: sqlite3.Connection, dataset_id: int, model: str) -> int:
    """Quantos discursos do dataset já têm embedding guardado para este modelo."""
    return conn.execute(
        "SELECT COUNT(*) FROM dataset_speeches ds JOIN embeddings e ON e.speech_id = ds.speech_id "
        "WHERE ds.dataset_id = ? AND e.model = ?",
        (dataset_id, model),
    ).fetchone()[0]


def save_embedding_speed(conn: sqlite3.Connection, model: str, device: str, sec_per_doc: float, n_docs: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embedding_speed (model, device, sec_per_doc, n_docs, measured_at) VALUES (?, ?, ?, ?, ?)",
        (model, device, sec_per_doc, n_docs, _now()),
    )
    conn.commit()


def get_embedding_speed(conn: sqlite3.Connection, model: str, device: str) -> dict | None:
    row = conn.execute("SELECT * FROM embedding_speed WHERE model=? AND device=?", (model, device)).fetchone()
    return dict(row) if row else None


def cached_month_totals(conn: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    """{'YYYY-MM': (total, fetched_at)} de todos os meses já consultados."""
    return {r["month"]: (r["total"], r["fetched_at"]) for r in conn.execute("SELECT * FROM plenary_month_totals")}


def save_month_total(conn: sqlite3.Connection, month: str, total: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO plenary_month_totals (month, total, fetched_at) VALUES (?, ?, ?)",
        (month, total, datetime.now().date().isoformat()),
    )
    conn.commit()


def create_run(conn: sqlite3.Connection, dataset_id: int, params: dict) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO runs (dataset_id, params, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (dataset_id, json.dumps(params), now, now),
    )
    conn.commit()
    return cur.lastrowid


def update_run(conn: sqlite3.Connection, run_id: int, **fields) -> None:
    if "summary" in fields and fields["summary"] is not None:
        fields["summary"] = json.dumps(fields["summary"], ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE runs SET {cols}, updated_at=? WHERE id=?", (*fields.values(), _now(), run_id))
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    row = conn.execute(
        "SELECT r.*, d.name AS dataset_name FROM runs r JOIN datasets d ON d.id = r.dataset_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    r = dict(row)
    r["params"] = json.loads(r["params"])
    r["summary"] = json.loads(r["summary"]) if r["summary"] else None
    return r


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    ids = [r["id"] for r in conn.execute("SELECT id FROM runs ORDER BY id DESC")]
    return [get_run(conn, i) for i in ids]


def delete_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()


def delete_dataset(conn: sqlite3.Connection, dataset_id: int) -> list[int]:
    """
    Apaga o dataset, as análises dele e os discursos que não pertencem a nenhum outro dataset.

    Returns:
        ids das análises apagadas (para remover os arquivos delas).
    """
    run_ids = [r["id"] for r in conn.execute("SELECT id FROM runs WHERE dataset_id=?", (dataset_id,))]
    conn.execute("DELETE FROM runs WHERE dataset_id=?", (dataset_id,))
    conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))  # cascades to dataset_speeches
    orphans = "SELECT id FROM speeches WHERE id NOT IN (SELECT speech_id FROM dataset_speeches)"
    conn.execute(f"DELETE FROM embeddings WHERE speech_id IN ({orphans})")
    conn.execute(f"DELETE FROM speeches WHERE id IN ({orphans})")
    conn.commit()
    return run_ids


def run_dir(run_id: int) -> Path:
    return RUNS_DIR / str(run_id)


def _dataset_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["keywords"] = json.loads(d["keywords"]) if d["keywords"] else []
    d["failed"] = d["failed"] or 0
    return d


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
