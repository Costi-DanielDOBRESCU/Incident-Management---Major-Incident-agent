"""
Starea tichetelor care nu apartin unui cluster, marcata de operator (SQLite, in audit.db).

Stari: 'handled' (tratat individual), 'watch' (de urmarit). 'unreviewed' e starea implicita si nu se stocheaza.
Cheia e (ticket_key, created): dupa 'Incepe un incident nou' cheile Jira se pot repeta, dar `created` difera,
deci o marcare veche nu se aplica din greseala unui tichet nou.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.execution import audit_store

STATUS_LABELS = {
    "unreviewed": "Nerevizuit",
    "handled": "Tratat individual",
    "watch": "De urmărit",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticket_reviews (
    ticket_key   TEXT NOT NULL,
    created      TEXT NOT NULL,
    status       TEXT NOT NULL,
    reviewed_by  TEXT NOT NULL,
    reviewed_at  TEXT NOT NULL,
    PRIMARY KEY (ticket_key, created)
);
"""


def _connect(db_path: Optional[Path] = None):
    conn = audit_store._connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def set_review(
    ticket_key: str,
    created: str,
    status: str,
    reviewed_by: str,
    db_path: Optional[Path] = None,
) -> None:
    """Salveaza (sau sterge, pentru 'unreviewed') starea unui tichet si scrie un eveniment de audit."""
    if status not in STATUS_LABELS:
        raise ValueError(f"Stare necunoscuta: {status!r}")

    with _connect(db_path) as conn:
        if status == "unreviewed":
            conn.execute(
                "DELETE FROM ticket_reviews WHERE ticket_key = ? AND created = ?", (ticket_key, created)
            )
        else:
            conn.execute(
                """
                INSERT INTO ticket_reviews (ticket_key, created, status, reviewed_by, reviewed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticket_key, created) DO UPDATE SET
                    status = excluded.status,
                    reviewed_by = excluded.reviewed_by,
                    reviewed_at = excluded.reviewed_at
                """,
                (ticket_key, created, status, reviewed_by, datetime.now(timezone.utc).isoformat()),
            )

    audit_store.log_audit_event(
        reviewed_by,
        "review_unclustered_ticket",
        input_ref=ticket_key,
        output_ref=status,
        based_on={"created": created},
        db_path=db_path,
    )


def list_reviews(db_path: Optional[Path] = None) -> dict[tuple[str, str], dict]:
    """Toate marcarile, indexate dupa (ticket_key, created)."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM ticket_reviews").fetchall()
    return {(r["ticket_key"], r["created"]): dict(r) for r in rows}