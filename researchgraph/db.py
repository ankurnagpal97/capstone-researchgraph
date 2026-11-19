"""SQLite storage: papers, accept/reject decisions, chunks, citation edges, query runs."""
from __future__ import annotations

import json
import sqlite3

from . import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,   -- arXiv ID without version, e.g. 2310.11511
    version       TEXT,               -- e.g. v1
    title         TEXT NOT NULL,
    authors       TEXT,               -- JSON list
    abstract      TEXT,
    published     TEXT,
    updated       TEXT,
    year          INTEGER,
    categories    TEXT,               -- JSON list
    doi           TEXT,
    openalex_id   TEXT,
    url           TEXT,
    license       TEXT,
    rights_status TEXT,               -- open_license | local_index_only | unknown_quarantine
    scope_tag     TEXT,
    pdf_path      TEXT,
    pdf_sha256    TEXT,
    n_pages       INTEGER,
    parse_status  TEXT,               -- pending | ok | failed: <reason>
    retrieved_at  TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    paper_id TEXT, decision TEXT, reason TEXT, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,     -- <paper_id>#<4-digit index>
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
    chunk_index INTEGER,
    section     TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    text        TEXT NOT NULL,
    n_words     INTEGER,
    est_tokens  INTEGER,
    text_sha256 TEXT,
    parser_version TEXT
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS citation_edges (
    citing_paper_id TEXT, cited_paper_id TEXT, edge_source TEXT, discovered_at TEXT,
    PRIMARY KEY (citing_paper_id, cited_paper_id, edge_source)
);
CREATE TABLE IF NOT EXISTS query_runs (
    run_id TEXT PRIMARY KEY, query TEXT, timestamp TEXT, config TEXT,
    selected_chunk_ids TEXT, graph_paper_ids TEXT, latency_ms INTEGER,
    llm_model TEXT, token_usage TEXT, prompt_version TEXT, answer_json TEXT
);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, table: str, row: dict) -> None:
    row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
