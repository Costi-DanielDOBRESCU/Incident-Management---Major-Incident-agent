"""
Retentia tichetelor din tabelul de ingestie (app/data/tickets.db).

Dupa o runda de detectie, tichetele care NU sunt legate de un incident major declarat nu se arunca
imediat: raman o vreme, ca sa fie reevaluate impreuna cu tichetele noi (un incident poate "creste").
Tichetele din clustere respinse / nepropuse / abandonate devin simple tichete individuale.

Reguli (prune_tickets):
  - legat de un incident major declarat (parent_incident_id setat)  -> se sterge
  - stare "Tratat individual" (handled)                             -> se sterge
  - stare "Nerevizuit" (unreviewed) / "De urmarit" (watch):
        ramane cat timp e mai nou de `retention_minutes`, apoi se sterge

Nu atinge watermark-ul de ingestie.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from app.ingestion import ticket_store

KEEP_STATUSES = ("unreviewed", "watch")


def prune_tickets(
    retention_minutes: int,
    reviews: Optional[dict[tuple[str, str], dict]] = None,
    *,
    now: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Args:
        retention_minutes: cat raman un tichet individual (nerevizuit / de urmarit).
        reviews: {(ticket_key, created): {"status": ...}} din ticket_reviews.list_reviews().
        now: epoch UTC (pentru teste).

    Returns:
        {"kept": [...chei], "dropped_linked": n, "dropped_handled": n, "dropped_expired": n}
    """
    reviews = reviews or {}
    now = time.time() if now is None else now
    cutoff = now - retention_minutes * 60

    kept: list[str] = []
    to_delete: list[str] = []
    counts = {"dropped_linked": 0, "dropped_handled": 0, "dropped_expired": 0}

    with ticket_store._connect(db_path) as conn:
        rows = conn.execute("SELECT key, created, created_ts, parent_incident_id FROM tickets").fetchall()
        for row in rows:
            key = row["key"]
            status = reviews.get((key, row["created"]), {}).get("status", "unreviewed")
            if row["parent_incident_id"]:
                counts["dropped_linked"] += 1
                to_delete.append(key)
            elif status == "handled":
                counts["dropped_handled"] += 1
                to_delete.append(key)
            elif status in KEEP_STATUSES and row["created_ts"] >= cutoff:
                kept.append(key)
            else:
                counts["dropped_expired"] += 1
                to_delete.append(key)

        conn.executemany("DELETE FROM tickets WHERE key = ?", [(k,) for k in to_delete])

    return {"kept": kept, **counts}