"""
Execution Layer (doc. sectiunea 4.3 / 6.4 pasul 7) - tool-uri DETERMINISTE.

Contract: LLM-ul propune, aici se executa DOAR ce a aprobat omul.
Nicio functie de aici nu apeleaza un LLM.

Tool-uri:
  - send_notification(...)        -> salveaza notificarea (SQLite + fisier .log), simulat
  - update_ticket_status(...)     -> status per tichet (tabel ticket_links)
  - link_tickets_to_parent(...)   -> leaga tichetele de incidentul-parinte
  - run_execution(decision)       -> orchestreaza cele de mai sus pornind de la
                                     decizia produsa de graph._build_decision()

Date in audit.db (acelasi fisier ca audit_store), tabele NOI - nu modifica tabelele existente:
  - notifications(incident_id, audience, subject, body, channel, status, sent_at)  [unic pe incident+audienta]
  - ticket_links(ticket_key, parent_incident_id, status, updated_at)
Notificarile se pot citi cu list_notifications() (pentru o pagina Streamlit ulterioara).

Fiecare actiune se auditeaza (actor "execution_layer") prin audit_store.log_audit_event.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.execution import audit_store

ACTOR = "execution_layer"
LINKED_STATUS = "Linked-MajorIncident"  # doc. pasul 10
VALID_AUDIENCES = ("end_users", "management")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    incident_id  TEXT NOT NULL,
    audience     TEXT NOT NULL,
    subject      TEXT NOT NULL,
    body         TEXT NOT NULL,
    channel      TEXT NOT NULL DEFAULT 'mock_email',
    status       TEXT NOT NULL DEFAULT 'sent',
    sent_at      TEXT NOT NULL,
    PRIMARY KEY (incident_id, audience)
);
CREATE INDEX IF NOT EXISTS idx_notifications_sent_at ON notifications(sent_at);

CREATE TABLE IF NOT EXISTS ticket_links (
    ticket_key          TEXT PRIMARY KEY,
    parent_incident_id  TEXT,
    status              TEXT,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticket_links_parent ON ticket_links(parent_incident_id);
"""


def _db(db_path: Optional[Path]) -> Path:
    return Path(db_path) if db_path else audit_store.DB_PATH


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = _db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_log_file(db_path: Optional[Path], line: str) -> None:
    """Jurnal text in plus, langa DB. Best effort: nu opreste executia."""
    try:
        log = _db(db_path).parent / "notifications.log"
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tool-uri
# ---------------------------------------------------------------------------
def send_notification(
    incident_id: str,
    audience: str,
    subject: str,
    body: str,
    *,
    channel: str = "mock_email",
    db_path: Optional[Path] = None,
) -> bool:
    """
    Trimite (simulat) o comunicare. Idempotent pe (incident_id, audience):
    a doua apelare NU mai trimite si intoarce False. True = trimisa acum.
    """
    if audience not in VALID_AUDIENCES:
        raise ValueError(f"Audienta necunoscuta: {audience!r}")

    sent_at = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO notifications "
            "(incident_id, audience, subject, body, channel, status, sent_at) "
            "VALUES (?, ?, ?, ?, ?, 'sent', ?)",
            (incident_id, audience, subject, body, channel, sent_at),
        )
        sent = cur.rowcount == 1

    if sent:
        _append_log_file(
            db_path,
            f"[{sent_at}] {channel} -> {audience} | {incident_id} | {subject}\n{body}\n{'-' * 60}",
        )
    audit_store.log_audit_event(
        ACTOR,
        "send_notification" if sent else "send_notification_skipped_duplicate",
        incident_id=incident_id,
        input_ref=f"audience:{audience}",
        output_ref=channel if sent else "already_sent",
        db_path=db_path,
    )
    return sent


def update_ticket_status(
    ticket_ids: list[str],
    status: str,
    *,
    incident_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Seteaza statusul fiecarui tichet (upsert, pastreaza parintele existent). Intoarce nr. tichete."""
    now = _now()
    with _connect(db_path) as conn:
        for key in ticket_ids:
            conn.execute(
                "INSERT INTO ticket_links (ticket_key, parent_incident_id, status, updated_at) "
                "VALUES (?, NULL, ?, ?) "
                "ON CONFLICT(ticket_key) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
                (key, status, now),
            )
    audit_store.log_audit_event(
        ACTOR,
        "update_ticket_status",
        incident_id=incident_id,
        input_ref=f"tickets:{','.join(ticket_ids)}",
        output_ref=status,
        db_path=db_path,
    )
    return len(ticket_ids)


def link_tickets_to_parent(
    ticket_ids: list[str],
    parent_incident_id: str,
    *,
    db_path: Optional[Path] = None,
) -> int:
    """Leaga tichetele de incidentul-parinte (upsert, pastreaza statusul existent). Intoarce nr. tichete."""
    now = _now()
    with _connect(db_path) as conn:
        for key in ticket_ids:
            conn.execute(
                "INSERT INTO ticket_links (ticket_key, parent_incident_id, status, updated_at) "
                "VALUES (?, ?, NULL, ?) "
                "ON CONFLICT(ticket_key) DO UPDATE SET "
                "parent_incident_id = excluded.parent_incident_id, updated_at = excluded.updated_at",
                (key, parent_incident_id, now),
            )
    audit_store.log_audit_event(
        ACTOR,
        "link_tickets_to_parent",
        incident_id=parent_incident_id,
        input_ref=f"tickets:{','.join(ticket_ids)}",
        output_ref=parent_incident_id,
        db_path=db_path,
    )
    return len(ticket_ids)


# ---------------------------------------------------------------------------
# Orchestrare
# ---------------------------------------------------------------------------
def run_execution(decision: dict[str, Any], *, db_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Executa actiunile aprobate. `decision` = dict-ul din graph._build_decision().
    - outcome != "declared"  -> nu executa nimic
    - notificari: doar audientele cu approved=True
    Intoarce un rezumat: {"executed", "notifications_sent", "notifications_skipped",
                          "tickets_linked", "reason"}.
    """
    result: dict[str, Any] = {
        "executed": False,
        "notifications_sent": [],
        "notifications_skipped": [],
        "tickets_linked": 0,
        "reason": None,
    }
    if decision.get("outcome") != "declared":
        result["reason"] = f"outcome={decision.get('outcome')}: nimic de executat"
        return result

    incident_id = decision["incident_id"]
    ticket_ids = list(decision.get("ticket_ids") or [])

    for audience, comm in (decision.get("communications") or {}).items():
        if not comm.get("approved"):
            result["notifications_skipped"].append(audience)
            continue
        if send_notification(incident_id, audience, comm["subject"], comm["body"], db_path=db_path):
            result["notifications_sent"].append(audience)
        else:
            result["notifications_skipped"].append(audience)  # deja trimisa anterior

    if ticket_ids:
        update_ticket_status(ticket_ids, LINKED_STATUS, incident_id=incident_id, db_path=db_path)
        result["tickets_linked"] = link_tickets_to_parent(ticket_ids, incident_id, db_path=db_path)

    result["executed"] = True
    return result


# ---------------------------------------------------------------------------
# Citire (pentru UI)
# ---------------------------------------------------------------------------
def list_notifications(
    limit: int = 100,
    incident_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Notificarile trimise, cele mai noi primele (optional filtrate pe incident)."""
    query = "SELECT * FROM notifications"
    params: list[Any] = []
    if incident_id:
        query += " WHERE incident_id = ?"
        params.append(incident_id)
    query += " ORDER BY sent_at DESC, rowid DESC LIMIT ?"
    params.append(limit)
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def list_ticket_links(
    parent_incident_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Tichetele legate / cu status setat (optional pe un incident)."""
    query = "SELECT * FROM ticket_links"
    params: list[Any] = []
    if parent_incident_id:
        query += " WHERE parent_incident_id = ?"
        params.append(parent_incident_id)
    query += " ORDER BY ticket_key"
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]