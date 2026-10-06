"""Tests for the SQLite audit log and the persisted incident decisions."""

import pytest

from app.execution import audit_store


@pytest.fixture
def db(tmp_path):
    return tmp_path / "audit.db"


def _decision(**overrides):
    base = {
        "incident_id": "MI-CL-1",
        "cluster_id": "CL-1",
        "service": "Internal Portal",
        "severity": "SEV2",
        "outcome": "declared",
        "reason": "",
        "decided_by": "alice",
        "decided_at": "2026-08-03T10:00:00+00:00",
        "ticket_ids": ["INC-1", "INC-2", "INC-3"],
        "ticket_count": 3,
        "summaries": ["Cannot log in", "SSO error"],
        "assessment": {"estimated_severity": "SEV2", "confidence": 0.9},
        "communications": {"end_users": {"subject": "S", "body": "B", "approved": True}},
    }
    base.update(overrides)
    return base


# ---------- audit log ----------

def test_log_event_roundtrip(db):
    event = audit_store.log_audit_event(
        "assessment_agent",
        "propose_major_incident",
        incident_id="MI-CL-1",
        input_ref="cluster:CL-1",
        output_ref="assessment:CL-1",
        model="openai/gpt-oss-120b (groq)",
        rag_sources=["PM-2025-0176", "RB-PORTAL-006"],
        based_on={"severity": "SEV2", "confidence": 0.95},
        db_path=db,
    )

    assert event["event_id"].startswith("evt_")

    stored = audit_store.list_audit_events(db_path=db)
    assert len(stored) == 1
    assert stored[0]["actor"] == "assessment_agent"
    assert stored[0]["action"] == "propose_major_incident"
    assert stored[0]["rag_sources"] == ["PM-2025-0176", "RB-PORTAL-006"]
    assert stored[0]["based_on"] == {"severity": "SEV2", "confidence": 0.95}
    assert stored[0]["model"] == "openai/gpt-oss-120b (groq)"


def test_events_are_listed_newest_first_and_filterable(db):
    audit_store.log_audit_event("a", "first", incident_id="MI-1", db_path=db)
    audit_store.log_audit_event("a", "second", incident_id="MI-2", db_path=db)
    audit_store.log_audit_event("a", "third", incident_id="MI-1", db_path=db)

    assert [e["action"] for e in audit_store.list_audit_events(db_path=db)] == ["third", "second", "first"]
    assert [e["action"] for e in audit_store.list_audit_events(incident_id="MI-1", db_path=db)] == ["third", "first"]
    assert len(audit_store.list_audit_events(limit=2, db_path=db)) == 2


def test_event_without_optional_fields(db):
    audit_store.log_audit_event("execution_layer", "noop", db_path=db)

    event = audit_store.list_audit_events(db_path=db)[0]

    assert event["incident_id"] is None
    assert event["rag_sources"] == []
    assert event["based_on"] is None


# ---------- decizii ----------

def test_save_and_list_decision(db):
    audit_store.save_decision(_decision(), db_path=db)

    decisions = audit_store.list_decisions(db_path=db)

    assert len(decisions) == 1
    d = decisions[0]
    assert d["incident_id"] == "MI-CL-1"
    assert d["outcome"] == "declared"
    assert d["ticket_ids"] == ["INC-1", "INC-2", "INC-3"]
    assert d["summaries"] == ["Cannot log in", "SSO error"]
    assert d["assessment"]["confidence"] == 0.9
    assert d["communications"]["end_users"]["approved"] is True


def test_save_decision_is_an_upsert(db):
    audit_store.save_decision(_decision(outcome="rejected", reason="duplicate"), db_path=db)
    audit_store.save_decision(_decision(outcome="declared", reason=""), db_path=db)

    decisions = audit_store.list_decisions(db_path=db)

    assert len(decisions) == 1
    assert decisions[0]["outcome"] == "declared"


def test_rejection_keeps_its_reason(db):
    audit_store.save_decision(_decision(outcome="rejected", reason="Licensing wave, not an outage", communications=None), db_path=db)

    d = audit_store.list_decisions(db_path=db)[0]

    assert d["outcome"] == "rejected"
    assert d["reason"] == "Licensing wave, not an outage"
    assert d["communications"] is None


def test_invalid_decisions_are_refused(db):
    with pytest.raises(ValueError):
        audit_store.save_decision(_decision(outcome="maybe"), db_path=db)
    with pytest.raises(ValueError):
        audit_store.save_decision(_decision(incident_id=""), db_path=db)


def test_decisions_listed_newest_first(db):
    audit_store.save_decision(_decision(incident_id="MI-A", decided_at="2026-08-03T09:00:00+00:00"), db_path=db)
    audit_store.save_decision(_decision(incident_id="MI-B", decided_at="2026-08-03T11:00:00+00:00"), db_path=db)

    assert [d["incident_id"] for d in audit_store.list_decisions(db_path=db)] == ["MI-B", "MI-A"]


def test_reset_clears_everything(db):
    audit_store.log_audit_event("a", "x", db_path=db)
    audit_store.save_decision(_decision(), db_path=db)

    audit_store.reset_audit_db(db_path=db)

    assert audit_store.list_audit_events(db_path=db) == []
    assert audit_store.list_decisions(db_path=db) == []