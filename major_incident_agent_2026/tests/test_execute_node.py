"""Teste pentru node_execute_actions (Execution Layer legat in graf)."""

from datetime import datetime, timezone

import pytest

from app.execution import audit_store, tools
from app.models.schemas import CommunicationDraft, IncidentCluster, MajorIncident
from app.orchestrator import graph


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_store, "DB_PATH", tmp_path / "audit.db")


def _state(approved_incident=True, approved_users=True, approved_mgmt=False):
    now = datetime.now(timezone.utc)
    cluster = IncidentCluster(
        cluster_id="CL-1",
        ticket_ids=["INC-1", "INC-2", "INC-3"],
        centroid_similarity=0.9,
        service_guess="VPN Gateway",
        window_start=now,
        window_end=now,
        ticket_count=3,
    )
    incident = MajorIncident(
        incident_id="MI-CL-1-abc123",
        cluster_id="CL-1",
        status="Declared" if approved_incident else "Rejected",
        severity="SEV2",
        declared_by="alice",
        declared_at=now,
    )
    drafts = {
        a: CommunicationDraft(incident_id="MI-CL-1-abc123", audience=a, subject=f"S-{a}", body=f"B-{a}")
        for a in ("end_users", "management")
    }
    return {
        "cluster": cluster,
        "incident_id": "MI-CL-1-abc123",
        "summaries": ["VPN down"],
        "major_incident": incident,
        "user_approved_incident": approved_incident,
        "communication_drafts": drafts,
        "user_approved_communications": {"end_users": approved_users, "management": approved_mgmt},
        "decided_by": "alice",
    }


def test_declared_incident_executes_only_approved_audiences():
    out = graph.node_execute_actions(_state())
    res = out["execution_result"]
    assert res["executed"] is True
    assert res["notifications_sent"] == ["end_users"]
    assert res["tickets_linked"] == 3
    assert [n["audience"] for n in tools.list_notifications()] == ["end_users"]


def test_rejected_incident_executes_nothing():
    out = graph.node_execute_actions(_state(approved_incident=False))
    assert out["execution_result"]["executed"] is False
    assert tools.list_notifications() == []


def test_node_does_not_touch_final_status():
    assert "final_status" not in graph.node_execute_actions(_state())


def test_execution_error_does_not_stop_the_flow(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(tools, "run_execution", boom)
    out = graph.node_execute_actions(_state())
    assert out["execution_result"]["executed"] is False
    assert "db down" in out["execution_result"]["error"]
    actions = [e["action"] for e in audit_store.list_audit_events()]
    assert "execute_actions_failed" in actions


def test_workflow_contains_execute_node_between_review_and_persist():
    builder = graph.create_workflow()
    assert "node_execute_actions" in builder.nodes
    edges = set(builder.edges)
    assert ("node_human_review_communication", "node_execute_actions") in edges
    assert ("node_execute_actions", "node_persist_and_learn") in edges
    assert ("node_human_review_communication", "node_persist_and_learn") not in edges