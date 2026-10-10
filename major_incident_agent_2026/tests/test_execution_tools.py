"""Teste pentru Execution Layer (notificari simulate, status + link tichete)."""

import pytest

from app.execution import audit_store, tools


@pytest.fixture
def db(tmp_path):
    return tmp_path / "audit.db"


def _decision(**overrides):
    base = {
        "incident_id": "MI-CL-1-abc123",
        "outcome": "declared",
        "ticket_ids": ["INC-1", "INC-2", "INC-3"],
        "communications": {
            "end_users": {"subject": "VPN issues", "body": "We are investigating.", "approved": True},
            "management": {"subject": "SEV2 VPN", "body": "3 tickets.", "approved": True},
        },
    }
    base.update(overrides)
    return base


def test_send_notification_is_saved_and_listed(db):
    assert tools.send_notification("MI-1", "end_users", "S", "B", db_path=db) is True
    rows = tools.list_notifications(db_path=db)
    assert len(rows) == 1
    assert rows[0]["audience"] == "end_users" and rows[0]["status"] == "sent"


def test_send_notification_writes_log_file(db):
    tools.send_notification("MI-1", "management", "Subj", "Body", db_path=db)
    log = (db.parent / "notifications.log").read_text(encoding="utf-8")
    assert "MI-1" in log and "Subj" in log


def test_send_notification_is_idempotent(db):
    assert tools.send_notification("MI-1", "end_users", "S", "B", db_path=db) is True
    assert tools.send_notification("MI-1", "end_users", "S2", "B2", db_path=db) is False
    assert len(tools.list_notifications(db_path=db)) == 1
    actions = [e["action"] for e in audit_store.list_audit_events(db_path=db)]
    assert "send_notification_skipped_duplicate" in actions


def test_invalid_audience_is_refused(db):
    with pytest.raises(ValueError):
        tools.send_notification("MI-1", "everyone", "S", "B", db_path=db)


def test_update_and_link_keep_each_others_fields(db):
    tools.link_tickets_to_parent(["INC-1"], "MI-1", db_path=db)
    tools.update_ticket_status(["INC-1"], "Linked-MajorIncident", db_path=db)
    row = tools.list_ticket_links(db_path=db)[0]
    assert row["parent_incident_id"] == "MI-1" and row["status"] == "Linked-MajorIncident"


def test_run_execution_declared_executes_everything(db):
    res = tools.run_execution(_decision(), db_path=db)
    assert res["executed"] is True
    assert sorted(res["notifications_sent"]) == ["end_users", "management"]
    assert res["tickets_linked"] == 3
    links = tools.list_ticket_links("MI-CL-1-abc123", db_path=db)
    assert len(links) == 3 and all(l["status"] == "Linked-MajorIncident" for l in links)


@pytest.mark.parametrize("outcome", ["rejected", "dismissed"])
def test_run_execution_does_nothing_unless_declared(db, outcome):
    res = tools.run_execution(_decision(outcome=outcome), db_path=db)
    assert res["executed"] is False
    assert tools.list_notifications(db_path=db) == []
    assert tools.list_ticket_links(db_path=db) == []


def test_unapproved_audience_is_not_sent(db):
    d = _decision()
    d["communications"]["management"]["approved"] = False
    res = tools.run_execution(d, db_path=db)
    assert res["notifications_sent"] == ["end_users"]
    assert res["notifications_skipped"] == ["management"]
    assert len(tools.list_notifications(db_path=db)) == 1


def test_run_execution_twice_does_not_duplicate(db):
    tools.run_execution(_decision(), db_path=db)
    res = tools.run_execution(_decision(), db_path=db)
    assert res["notifications_sent"] == []
    assert len(tools.list_notifications(db_path=db)) == 2


def test_everything_is_audited(db):
    tools.run_execution(_decision(), db_path=db)
    actions = {e["action"] for e in audit_store.list_audit_events(db_path=db)}
    assert {"send_notification", "update_ticket_status", "link_tickets_to_parent"} <= actions