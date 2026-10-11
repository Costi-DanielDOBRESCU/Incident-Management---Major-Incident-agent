"""Teste pentru Execution Layer (notificari simulate, status + link tichete)."""

import pytest

from app.execution import audit_store, tools
from app.ingestion import ticket_store


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


def _seed_tickets(db, keys):
    """Creeaza tickets.db langa audit.db (asa il foloseste run_execution in teste)."""
    tdb = db.parent / "tickets.db"
    ticket_store.init_db(tdb)
    issues = [
        {
            "key": k,
            "fields": {
                "summary": "s", "description": "d", "created": "2026-10-10T10:00:00.000+0000",
                "priority": {"name": "P2"}, "status": {"name": "Open"},
                "components": [{"name": "VPN Gateway"}],
            },
        }
        for k in keys
    ]
    ticket_store.upsert_tickets(issues, db_path=tdb)
    return tdb


def test_run_execution_writes_link_and_status_into_ticket_store(db):
    tdb = _seed_tickets(db, ["INC-1", "INC-2", "INC-3", "INC-9"])
    res = tools.run_execution(_decision(cluster_id="CL-7"), db_path=db)
    assert res["tickets_synced"] == 3
    rows = {r["key"]: r for r in ticket_store.list_tickets(db_path=tdb)}
    assert rows["INC-1"]["parent_incident_id"] == "MI-CL-1-abc123"
    assert rows["INC-1"]["ticket_status"] == "Linked-MajorIncident"
    assert rows["INC-1"]["cluster_id"] == "CL-7"
    assert rows["INC-9"]["parent_incident_id"] is None  # tichet din afara clusterului: neatins


def test_rejected_incident_does_not_touch_ticket_store(db):
    tdb = _seed_tickets(db, ["INC-1"])
    tools.run_execution(_decision(outcome="rejected"), db_path=db)
    assert ticket_store.list_tickets(db_path=tdb)[0]["parent_incident_id"] is None


def test_ticket_store_failure_does_not_stop_execution(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("tickets db locked")

    monkeypatch.setattr(ticket_store, "link_tickets", boom)
    res = tools.run_execution(_decision(), db_path=db)
    assert res["executed"] is True and res["tickets_synced"] == 0
    actions = [e["action"] for e in audit_store.list_audit_events(db_path=db)]
    assert "sync_ticket_store_failed" in actions


def test_notification_is_written_as_eml_in_outbox(db):
    import email
    from email import policy

    tools.send_notification("MI-1", "management", "Subiect test", "Corp mesaj", db_path=db)
    path = tools.outbox_path("MI-1", "management", db_path=db)
    assert path.exists() and path.parent.name == "outbox"
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    assert msg["To"] == tools.RECIPIENTS["management"]
    assert msg["From"] == tools.SENDER
    assert msg["Subject"] == "Subiect test"
    assert msg["X-MIA-Incident"] == "MI-1"
    assert "Corp mesaj" in msg.get_content()


def test_duplicate_send_does_not_rewrite_eml(db):
    tools.send_notification("MI-1", "end_users", "Prima", "B", db_path=db)
    tools.send_notification("MI-1", "end_users", "A doua", "B2", db_path=db)
    assert b"Prima" in tools.outbox_path("MI-1", "end_users", db_path=db).read_bytes()


def test_unapproved_audience_gets_no_eml(db):
    d = _decision()
    d["communications"]["management"]["approved"] = False
    tools.run_execution(d, db_path=db)
    assert tools.outbox_path("MI-CL-1-abc123", "end_users", db_path=db).exists()
    assert not tools.outbox_path("MI-CL-1-abc123", "management", db_path=db).exists()