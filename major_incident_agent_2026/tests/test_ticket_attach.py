"""Teste pentru atasarea manuala a unui tichet la un incident declarat."""

import pytest

from app.execution import audit_store, ticket_archive, ticket_attach, tools
from app.ingestion import ticket_store


@pytest.fixture
def dbs(tmp_path):
    audit, tickets = tmp_path / "audit.db", tmp_path / "tickets.db"
    ticket_store.init_db(tickets)
    issues = [
        {
            "key": k,
            "fields": {
                "summary": f"s {k}", "description": "d", "created": "2026-10-10T10:00:00.000+0000",
                "priority": {"name": "P2"}, "status": {"name": "Open"}, "components": [{"name": "VPN Gateway"}],
            },
        }
        for k in ("INC-1", "INC-2", "LATE-1")
    ]
    ticket_store.upsert_tickets(issues, db_path=tickets)
    return audit, tickets


def _save(audit, outcome="declared", incident_id="MI-1"):
    audit_store.save_decision(
        {
            "incident_id": incident_id, "cluster_id": "CL-1", "service": "VPN", "severity": "SEV2",
            "outcome": outcome, "decided_by": "alice", "ticket_ids": ["INC-1", "INC-2"], "ticket_count": 2,
            "summaries": ["a", "b"], "assessment": None, "communications": None,
        },
        db_path=audit,
    )


def _attach(dbs, key="LATE-1", incident="MI-1"):
    audit, tickets = dbs
    return ticket_attach.attach_ticket_to_incident(key, incident, "bob", audit_db_path=audit, tickets_db_path=tickets)


def test_attach_updates_everything(dbs):
    audit, tickets = dbs
    _save(audit)
    res = _attach(dbs)
    assert res["ticket_count"] == 3

    decision = audit_store.list_decisions(db_path=audit)[0]
    assert decision["ticket_ids"] == ["INC-1", "INC-2", "LATE-1"] and decision["ticket_count"] == 3
    assert decision["summaries"] == ["a", "b"]  # restul deciziei ramane neschimbat

    row = {r["key"]: r for r in ticket_store.list_tickets(db_path=tickets)}["LATE-1"]
    assert row["parent_incident_id"] == "MI-1" and row["cluster_id"] == "CL-1"
    assert [l["ticket_key"] for l in tools.list_ticket_links("MI-1", db_path=audit)] == ["LATE-1"]
    assert set(ticket_archive.get_incident_tickets("MI-1", db_path=audit)) == {"LATE-1"}

    events = audit_store.list_audit_events(incident_id="MI-1", db_path=audit)
    assert any(e["action"] == "attach_ticket_manual" and e["actor"] == "bob" for e in events)


def test_attached_ticket_is_excluded_from_next_detection(dbs):
    audit, tickets = dbs
    _save(audit)
    _attach(dbs)
    keys = [i["key"] for i in ticket_store.list_raw_issues(db_path=tickets, exclude_linked=True)]
    assert "LATE-1" not in keys


def test_cannot_attach_to_rejected_incident(dbs):
    audit, _ = dbs
    _save(audit, outcome="rejected")
    with pytest.raises(ValueError, match="declarat"):
        _attach(dbs)


def test_unknown_incident_and_unknown_ticket(dbs):
    audit, _ = dbs
    _save(audit)
    with pytest.raises(ValueError, match="nu există"):
        _attach(dbs, incident="MI-NOPE")
    with pytest.raises(ValueError, match="nu mai există"):
        _attach(dbs, key="GHOST")


def test_cannot_attach_twice_or_a_ticket_already_linked(dbs):
    audit, _ = dbs
    _save(audit)
    _attach(dbs)
    with pytest.raises(ValueError, match="deja legat"):
        _attach(dbs)
    assert audit_store.list_decisions(db_path=audit)[0]["ticket_count"] == 3  # nu s-a dublat


def test_list_declared_incidents_filters_outcome(dbs):
    audit, _ = dbs
    _save(audit, incident_id="MI-1")
    _save(audit, outcome="rejected", incident_id="MI-2")
    assert [d["incident_id"] for d in ticket_attach.list_declared_incidents(db_path=audit)] == ["MI-1"]