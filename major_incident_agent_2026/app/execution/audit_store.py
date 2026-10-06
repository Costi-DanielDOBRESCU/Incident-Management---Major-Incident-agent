"""
app/execution/audit_store.py

Persistare pentru Etapa 8 (doc. sectiunea 8.2, "Audit trail"), in SQLite local: app/data/audit.db
(acoperit de `*.db` din .gitignore).

Doua tabele:
  - audit_events:       jurnal de audit append-only; populeaza AuditLogEntry (schemas.py).
                        Cine (actor), ce (action), cand, pe ce baza (rag_sources, based_on), cu ce model.
  - incident_decisions: cate un rand per incident DECIS DE UN OM (declarat sau respins), cu motivul,
                        clusterul, evaluarea si comunicarile. Sursa pentru "Istoric decizii" din UI
                        si pentru memoria de incidente (incident_memory.py).

Separat intentionat de tickets.db: tichetele se reseteaza la fiecare incident nou, auditul NU.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# app/execution/audit_store.py -> app/execution -> app -> app/data/audit.db
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id     TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    incident_id  TEXT,
    input_ref    TEXT,
    output_ref   TEXT,
    model        TEXT,
    rag_sources  TEXT NOT NULL DEFAULT '[]',
    based_on     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_incident  ON audit_events(incident_id);

CREATE TABLE IF NOT EXISTS incident_decisions (
    incident_id    TEXT PRIMARY KEY,
    cluster_id     TEXT NOT NULL,
    service        TEXT,
    severity       TEXT,
    outcome        TEXT NOT NULL,      -- declared | rejected
    reason         TEXT,
    decided_by     TEXT,
    decided_at     TEXT,
    ticket_ids     TEXT NOT NULL DEFAULT '[]',
    ticket_count   INTEGER,
    summaries      TEXT NOT NULL DEFAULT '[]',
    assessment     TEXT,               -- JSON
    communications TEXT                -- JSON: {audience: {subject, body, rag_sources, approved}}
);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON incident_decisions(decided_at);
"""

_JSON_DECISION_FIELDS = ("ticket_ids", "summaries", "assessment", "communications")


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def log_audit_event(
    actor: str,
    action: str,
    *,
    incident_id: Optional[str] = None,
    input_ref: Optional[str] = None,
    output_ref: Optional[str] = None,
    model: Optional[str] = None,
    rag_sources: Optional[list[str]] = None,
    based_on: Optional[dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Scrie o intrare in jurnalul de audit si o intoarce (cu event_id si timestamp)."""
    event = {
        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
        "timestamp": _now(),
        "actor": actor,
        "action": action,
        "incident_id": incident_id,
        "input_ref": input_ref,
        "output_ref": output_ref,
        "model": model,
        "rag_sources": list(rag_sources or []),
        "based_on": based_on,
    }
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_events
                (event_id, timestamp, actor, action, incident_id, input_ref, output_ref, model, rag_sources, based_on)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"], event["timestamp"], actor, action, incident_id, input_ref, output_ref, model,
                json.dumps(event["rag_sources"], ensure_ascii=False),
                json.dumps(based_on, ensure_ascii=False, default=str) if based_on is not None else None,
            ),
        )
    return event


def list_audit_events(
    limit: int = 100,
    incident_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Ultimele evenimente de audit, cele mai noi primele (optional filtrate pe incident)."""
    query = "SELECT * FROM audit_events"
    params: list[Any] = []
    if incident_id:
        query += " WHERE incident_id = ?"
        params.append(incident_id)
    query += " ORDER BY timestamp DESC, rowid DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    events = []
    for row in rows:
        event = dict(row)
        event["rag_sources"] = json.loads(event["rag_sources"] or "[]")
        event["based_on"] = json.loads(event["based_on"]) if event["based_on"] else None
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Decizii pe incident
# ---------------------------------------------------------------------------
def save_decision(decision: dict[str, Any], db_path: Optional[Path] = None) -> None:
    """
    Upsert dupa incident_id. Campuri obligatorii: incident_id, cluster_id, outcome ('declared'|'rejected').
    Campurile JSON (ticket_ids, summaries, assessment, communications) se primesc ca obiecte Python.
    """
    if decision.get("outcome") not in ("declared", "rejected"):
        raise ValueError("outcome trebuie sa fie 'declared' sau 'rejected'.")
    for required in ("incident_id", "cluster_id"):
        if not decision.get(required):
            raise ValueError(f"Campul '{required}' este obligatoriu.")

    row = {
        "incident_id": decision["incident_id"],
        "cluster_id": decision["cluster_id"],
        "service": decision.get("service"),
        "severity": decision.get("severity"),
        "outcome": decision["outcome"],
        "reason": decision.get("reason"),
        "decided_by": decision.get("decided_by"),
        "decided_at": decision.get("decided_at") or _now(),
        "ticket_ids": json.dumps(decision.get("ticket_ids") or [], ensure_ascii=False),
        "ticket_count": decision.get("ticket_count"),
        "summaries": json.dumps(decision.get("summaries") or [], ensure_ascii=False),
        "assessment": json.dumps(decision["assessment"], ensure_ascii=False, default=str) if decision.get("assessment") is not None else None,
        "communications": json.dumps(decision["communications"], ensure_ascii=False, default=str) if decision.get("communications") is not None else None,
    }
    columns = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k} = excluded.{k}" for k in row if k != "incident_id")

    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO incident_decisions ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(incident_id) DO UPDATE SET {updates}",
            row,
        )


def list_decisions(limit: int = 50, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Ultimele decizii umane, cele mai noi primele; campurile JSON sunt decodate."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM incident_decisions ORDER BY decided_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()

    decisions = []
    for row in rows:
        decision = dict(row)
        for field in _JSON_DECISION_FIELDS:
            decision[field] = json.loads(decision[field]) if decision[field] else None
        decisions.append(decision)
    return decisions


def reset_audit_db(db_path: Optional[Path] = None) -> None:
    """Sterge TOT auditul si deciziile (doar pentru demo/teste)."""
    with _connect(db_path) as conn:
        conn.executescript("DROP TABLE IF EXISTS audit_events; DROP TABLE IF EXISTS incident_decisions;")
    _connect(db_path).close()


# ---------------------------------------------------------------------------
# CLI: python -m app.execution.audit_store [--limit 30] [--incident MI-...] [--decisions]
# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description="Vizualizeaza jurnalul de audit si deciziile persistate.")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--incident", default=None, help="filtreaza evenimentele dupa incident_id")
    ap.add_argument("--decisions", action="store_true", help="afiseaza deciziile pe incident, nu evenimentele")
    args = ap.parse_args()

    if args.decisions:
        rows = list_decisions(args.limit)
        print(f"{len(rows)} decizii (cele mai noi primele)\n")
        for d in rows:
            print(f"{d['decided_at']}  {d['incident_id']:<28}{d['outcome']:<10}{d['severity'] or '-':<6}"
                  f"{d['service'] or '-':<18}by {d['decided_by'] or '-'}")
            if d["reason"]:
                print(f"    motiv: {d['reason']}")
        return

    rows = list_audit_events(args.limit, args.incident)
    print(f"{len(rows)} evenimente (cele mai noi primele)\n")
    for e in rows:
        sources = ",".join(e["rag_sources"]) if e["rag_sources"] else "-"
        print(f"{e['timestamp']}  {e['actor']:<22}{e['action']:<28}{e['incident_id'] or '-':<26}model={e['model'] or '-'}  rag={sources}")


if __name__ == "__main__":
    _main()