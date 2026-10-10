"""
Retentia tichetelor fara cluster (tabelul de ingestie, app/data/tickets.db).

Dupa o runda de detectie, tichetele care NU au intrat in niciun cluster nu se arunca imediat:
raman in tabel o vreme, ca sa fie reevaluate impreuna cu tichetele noi care sosesc intre timp
(un incident poate "creste" si atunci tichetele vechi, apropiate de prag, se pot alipi).

Reguli (prune_tickets):
  - tichete aflate intr-un cluster al rundei incheiate  -> se sterg (incidentul a fost tratat/abandonat)
  - stare "Tratat individual" (handled)                  -> se sterg
  - stare "Nerevizuit" (unreviewed) / "De urmarit" (watch), fara cluster:
        raman cat timp sunt mai noi de `retention_minutes`, apoi se sterg

Nu modifica ticket_store.py in afara de citire/stergere si nu atinge watermark-ul de ingestie.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Optional

from app.ingestion import ticket_store

KEEP_STATUSES = ("unreviewed", "watch")


def prune_tickets(
    clustered_keys: Iterable[str],
    retention_minutes: int,
    reviews: Optional[dict[tuple[str, str], dict]] = None,
    *,
    now: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Sterge din tabelul de ingestie tichetele care nu mai trebuie pastrate.

    Args:
        clustered_keys: cheile tichetelor din clusterele rundei incheiate.
        retention_minutes: cat ramane un tichet fara cluster (nerevizuit / de urmarit).
        reviews: {(ticket_key, created): {"status": ...}} din ticket_reviews.list_reviews().
        now: epoch UTC (pentru teste).

    Returns:
        {"kept": [...chei], "dropped_clustered": n, "dropped_handled": n, "dropped_expired": n}
    """
    reviews = reviews or {}
    clustered = set(clustered_keys)
    now = time.time() if now is None else now
    cutoff = now - retention_minutes * 60

    kept: list[str] = []
    to_delete: list[str] = []
    counts = {"dropped_clustered": 0, "dropped_handled": 0, "dropped_expired": 0}

    with ticket_store._connect(db_path) as conn:
        rows = conn.execute("SELECT key, created, created_ts FROM tickets").fetchall()
        for row in rows:
            key = row["key"]
            status = reviews.get((key, row["created"]), {}).get("status", "unreviewed")
            if key in clustered:
                counts["dropped_clustered"] += 1
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