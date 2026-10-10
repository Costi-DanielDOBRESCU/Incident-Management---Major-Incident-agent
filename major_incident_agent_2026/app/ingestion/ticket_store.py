"""
app/ingestion/ticket_store.py

Tabel local (SQLite) cu tichetele ingerate din Jira (mock sau real).
Folosit de: ingestor (scrie), Streamlit (citeste), detector (citeste, pasul urmator).

Fisier DB: app/data/tickets.db  (acoperit de `*.db` din .gitignore)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# app/ingestion/ticket_store.py -> app/ingestion -> app -> app/data/tickets.db
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    key          TEXT PRIMARY KEY,
    created      TEXT NOT NULL,      -- ISO, exact ca in Jira
    created_ts   REAL NOT NULL,      -- epoch UTC, pentru ferestre de timp
    service      TEXT,
    priority     TEXT,
    summary      TEXT,
    description  TEXT,
    raw_json     TEXT NOT NULL,      -- issue-ul Jira complet
    fetched_at   TEXT NOT NULL,
    cluster_id   TEXT                -- NULL pana e grupat de detector
);
CREATE INDEX IF NOT EXISTS idx_tickets_created_ts ON tickets(created_ts);

CREATE TABLE IF NOT EXISTS ingest_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    watermark  TEXT                  -- `created` al ultimului tichet ingerat
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Streamlit poate citi cat timp ingestorul scrie
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO ingest_state (id, watermark) VALUES (1, NULL)")


def reset_db(db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript("DROP TABLE IF EXISTS tickets; DROP TABLE IF EXISTS ingest_state;")
    init_db(db_path)


def parse_jira_ts(value: str) -> float:
    """'2026-09-30T14:06:16.711+0000' (sau cu 'Z') -> epoch UTC."""
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+0000"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(v, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Format de data necunoscut: {value!r}")


def get_watermark(db_path: Optional[Path] = None) -> Optional[str]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT watermark FROM ingest_state WHERE id = 1").fetchone()
    return row["watermark"] if row else None


def upsert_tickets(issues: list[dict[str, Any]], db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """
    Insereaza tichetele noi (dedup dupa `key`). Intoarce DOAR tichetele care erau noi.
    Actualizeaza si watermark-ul la cel mai recent `created` vazut.
    """
    if not issues:
        return []

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")
    new_issues: list[dict[str, Any]] = []
    newest_created: Optional[str] = None
    newest_ts = float("-inf")

    with _connect(db_path) as conn:
        for issue in issues:
            f = issue["fields"]
            created = f["created"]
            ts = parse_jira_ts(created)
            components = f.get("components") or []
            service = components[0]["name"] if components else "Unknown"
            priority = (f.get("priority") or {}).get("name", "")

            # Aceeasi cheie Jira cu alt `created` = tichet NOU (cheile se repeta dupa un reset al mock-ului):
            # il inlocuieste pe cel vechi, ca un tichet pastrat din runda anterioara sa nu-l blocheze.
            prev = conn.execute("SELECT created FROM tickets WHERE key = ?", (issue["key"],)).fetchone()
            if prev is not None and prev["created"] != created:
                conn.execute("DELETE FROM tickets WHERE key = ?", (issue["key"],))

            cur = conn.execute(
                """
                INSERT OR IGNORE INTO tickets
                    (key, created, created_ts, service, priority, summary, description, raw_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    issue["key"], created, ts, service, priority,
                    f.get("summary", ""), f.get("description", ""),
                    json.dumps(issue, ensure_ascii=False), fetched_at,
                ),
            )
            if cur.rowcount:
                new_issues.append(issue)
            if ts > newest_ts:
                newest_ts, newest_created = ts, created

        if newest_created is not None:
            conn.execute("UPDATE ingest_state SET watermark = ? WHERE id = 1", (newest_created,))

    return new_issues


def list_tickets(limit: int = 200, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Ultimele `limit` tichete, cele mai noi primele."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key, created, service, priority, summary, cluster_id "
            "FROM tickets ORDER BY created_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_raw_issues(limit: int = 1000, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Issue-urile Jira complete, in ordine cronologica (pentru detectie/clustering)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT raw_json FROM tickets ORDER BY created_ts ASC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["raw_json"]) for r in rows]


def count_tickets(db_path: Optional[Path] = None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]

def get_raw_issue(key: str, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Issue-ul Jira complet pentru un tichet (None daca nu exista)."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT raw_json FROM tickets WHERE key = ?", (key,)).fetchone()
    return json.loads(row["raw_json"]) if row else None