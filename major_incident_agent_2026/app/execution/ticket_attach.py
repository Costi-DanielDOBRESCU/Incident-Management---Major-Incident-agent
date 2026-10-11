"""
Atasare MANUALA a unui tichet la un incident major deja declarat (actiune a operatorului, din Istoric).

Cazul real: un tichet sosit tarziu sau clasificat gresit de detector apartine, de fapt, unui incident declarat.

Ce actualizeaza (toate best-effort coerente: se valideaza intai, apoi se scrie):
  - incident_decisions (audit.db): ticket_ids + ticket_count
  - ticket_links (audit.db) si tickets.db: legatura tichet -> incident, status, cluster_id
  - arhiva incidentului (incident_tickets): issue-ul complet, ca sa se vada in Istoric
  - audit_events: "attach_ticket_manual", actor = operatorul

Nu apeleaza niciun LLM. Se atasaza doar la incidente cu outcome "declared".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.execution import audit_store, ticket_archive, tools
from app.ingestion import ticket_store


def list_declared_incidents(limit: int = 500, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Incidentele declarate (cele mai noi primele): singurele la care se poate atasa un tichet."""
    return [d for d in audit_store.list_decisions(limit=limit, db_path=db_path) if d["outcome"] == "declared"]


def attach_ticket_to_incident(
    ticket_key: str,
    incident_id: str,
    attached_by: str,
    *,
    audit_db_path: Optional[Path] = None,
    tickets_db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Ataseaza tichetul la incident. Ridica ValueError cu un mesaj clar daca actiunea nu e valida
    (incident inexistent / nedeclarat, tichet inexistent sau deja legat de un incident).
    Intoarce {"incident_id", "ticket_key", "ticket_count"}.
    """
    decision = next(
        (d for d in audit_store.list_decisions(limit=1000, db_path=audit_db_path) if d["incident_id"] == incident_id),
        None,
    )
    if decision is None:
        raise ValueError(f"Incidentul {incident_id} nu există.")
    if decision["outcome"] != "declared":
        raise ValueError("Se poate atașa un tichet doar la un incident declarat.")

    issue = ticket_store.get_raw_issue(ticket_key, tickets_db_path)
    if issue is None:
        raise ValueError(f"Tichetul {ticket_key} nu mai există în baza locală.")

    rows = {r["key"]: r for r in ticket_store.list_tickets(limit=5000, db_path=tickets_db_path)}
    parent = (rows.get(ticket_key) or {}).get("parent_incident_id")
    ticket_ids = list(decision.get("ticket_ids") or [])
    if parent or ticket_key in ticket_ids:
        raise ValueError(f"Tichetul {ticket_key} este deja legat de un incident ({parent or incident_id}).")

    # 1) decizia: lista de tichete + numarul lor
    ticket_ids.append(ticket_key)
    decision["ticket_ids"] = ticket_ids
    decision["ticket_count"] = len(ticket_ids)
    audit_store.save_decision(decision, db_path=audit_db_path)

    # 2) legaturile (audit.db si tickets.db) + arhiva
    tools.link_tickets_to_parent([ticket_key], incident_id, db_path=audit_db_path)
    tools.update_ticket_status([ticket_key], tools.LINKED_STATUS, incident_id=incident_id, db_path=audit_db_path)
    ticket_store.link_tickets(
        [ticket_key], incident_id, tools.LINKED_STATUS, cluster_id=decision.get("cluster_id"), db_path=tickets_db_path
    )
    ticket_archive.archive_incident_tickets(incident_id, [issue], db_path=audit_db_path)

    # 3) audit
    audit_store.log_audit_event(
        attached_by,
        "attach_ticket_manual",
        incident_id=incident_id,
        input_ref=ticket_key,
        output_ref=f"{len(ticket_ids)} tickets",
        based_on={"manual": True},
        db_path=audit_db_path,
    )
    return {"incident_id": incident_id, "ticket_key": ticket_key, "ticket_count": len(ticket_ids)}