"""
Arhiva tichetelor unui incident evaluat (audit.db, tabel `incident_tickets`).

De ce: tabelul de ingestie (tickets.db) se curata (retentie, "Incepe un incident nou", chei Jira care se
repeta), deci Istoricul nu isi mai poate vedea tichetele. La salvarea deciziei, copiem issue-urile Jira
complete ale clusterului aici, legate de incident. Istoricul citeste din arhiva, nu din tickets.db.

Foloseste aceeasi conexiune ca audit_store (la fel ca ticket_reviews); nu modifica tabelele existente.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.execution import audit_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incident_tickets (
    incident_id  TEXT NOT NULL,
    ticket_key   TEXT NOT NULL,
    raw_json     TEXT NOT NULL,      -- issue-ul Jira complet, ca la momentul deciziei
    archived_at  TEXT NOT NULL,
    PRIMARY KEY (incident_id, ticket_key)
);
"""


def _connect(db_path: Optional[Path] = None):
    conn = audit_store._connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def archive_incident_tickets(
    incident_id: str, issues: list[dict[str, Any]], db_path: Optional[Path] = None
) -> int:
    """Salveaza (upsert) issue-urile unui incident. Intoarce cate tichete au fost arhivate."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        for issue in issues:
            conn.execute(
                "INSERT OR REPLACE INTO incident_tickets (incident_id, ticket_key, raw_json, archived_at) "
                "VALUES (?, ?, ?, ?)",
                (incident_id, issue["key"], json.dumps(issue, ensure_ascii=False), now),
            )
    return len(issues)


def get_incident_tickets(incident_id: str, db_path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """{ticket_key: issue Jira complet} pentru un incident (gol daca nu exista arhiva)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ticket_key, raw_json FROM incident_tickets WHERE incident_id = ?", (incident_id,)
        ).fetchall()
    return {r["ticket_key"]: json.loads(r["raw_json"]) for r in rows}