"""Teste pentru nodurile grafului: audit, decizie umana, persistenta si id-uri unice (fara LLM / Chroma reale)."""

from datetime import datetime, timezone

import pytest

from app.execution import audit_store, incident_memory
from app.models.schemas import CommunicationDraft, IncidentAssessment, IncidentCluster
from app.orchestrator import graph as g


def _cluster(cluster_id="CL-20261006-001"):
    now = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
    return IncidentCluster(
        cluster_id=cluster_id,
        ticket_ids=["T-1", "T-2", "T-3"],
        centroid_similarity=0.9,
        service_guess="Network/Switch",
        window_start=now,
        window_end=now,
        ticket_count=3,
    )


def _assessment(candidate=True, severity="SEV2", cluster_id="CL-20261006-001"):
    return IncidentAssessment(
        cluster_id=cluster_id,
        is_major_incident_candidate=candidate,
        confidence=0.9,
        estimated_severity=severity,
        affected_service="Network/Switch",
        reasoning="Switch down in building A",
        rag_sources=["RB-NET-001"],
        recommended_action="propose_major_incident" if candidate else "dismiss",
    )


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """audit.db temporar; LLM-ul si eticheta modelului inlocuite; memoria Chroma inlocuita."""
    monkeypatch.setattr(audit_store, "DB_PATH", tmp_path / "audit.db")
    monkeypatch.setattr(g, "last_llm_label", lambda: "test-model (stub)")
    monkeypatch.setattr(g, "assess_incident", lambda cluster, summaries: _assessment(cluster_id=cluster.cluster_id))

    remembered = []
    monkeypatch.setattr(incident_memory, "remember_decision", lambda d: remembered.append(d) or "MEM-1")
    return remembered


def _assessed_state(cluster=None, candidate=True):
    cluster = cluster or _cluster()
    state = {"cluster": cluster.model_dump(mode="json"), "summaries": ["Switch down", "No network", "Timeouts"]}
    state.update(g.node_assess_incident(state))
    state["assessment"] = _assessment(candidate=candidate, cluster_id=cluster.cluster_id)
    return state


def _review(state, monkeypatch, resume):
    monkeypatch.setattr(g, "interrupt", lambda payload: resume)
    return g.node_human_review_incident(state)


# ---------- assess ----------

def test_assess_sets_unique_incident_id_and_audits():
    first = g.node_assess_incident({"cluster": _cluster(), "summaries": []})
    second = g.node_assess_incident({"cluster": _cluster(), "summaries": []})

    assert first["incident_id"].startswith("MI-CL-20261006-001-")
    assert first["incident_id"] != second["incident_id"]

    events = audit_store.list_audit_events(incident_id=first["incident_id"])
    assert len(events) == 1
    assert events[0]["action"] == "propose_major_incident"
    assert events[0]["model"] == "test-model (stub)"
    assert events[0]["rag_sources"] == ["RB-NET-001"]


def _not_candidate_state(monkeypatch):
    monkeypatch.setattr(
        g, "assess_incident",
        lambda c, s: _assessment(candidate=False, severity="SEV3", cluster_id=c.cluster_id),
    )
    state = {"cluster": _cluster(), "summaries": ["mailbox full", "need larger mailbox", "over quota"]}
    state.update(g.node_assess_incident(state))
    return state


def test_not_candidate_is_audited_and_goes_to_human_review(monkeypatch):
    state = _not_candidate_state(monkeypatch)

    assert audit_store.list_audit_events(incident_id=state["incident_id"])[0]["action"] == "assess_not_candidate"
    # orice cluster evaluat trece prin om, chiar daca Assessment nu l-a propus
    assert ("node_assess_incident", "node_human_review_incident") in g.create_workflow().edges


def test_human_confirms_not_incident(isolated, monkeypatch):
    state = _not_candidate_state(monkeypatch)
    out = _review(state, monkeypatch, {"approved": False, "reason": "mailbox quota", "decided_by": "alice"})

    assert out["major_incident"].status == "Rejected"
    assert out["major_incident"].severity == "SEV3"
    assert out["rejection_reason"] == "mailbox quota"
    assert g.route_after_incident_review({**state, **out}) == "node_persist_and_learn"

    events = audit_store.list_audit_events(incident_id=state["incident_id"])
    review = next(e for e in events if e["action"] == "confirm_not_incident")
    assert review["actor"] == "alice" and review["based_on"]["ai_candidate"] is False

    state.update(out)
    g.node_persist_and_learn(state)
    (decision,) = audit_store.list_decisions()
    assert decision["outcome"] == "rejected"
    assert decision["assessment"]["is_major_incident_candidate"] is False  # din asta se deduce "AI nu a propus"
    assert len(isolated) == 1 and isolated[0]["outcome"] == "rejected"  # decizie umana -> memorie


def test_human_declares_incident_that_ai_did_not_propose(monkeypatch):
    state = _not_candidate_state(monkeypatch)
    out = _review(state, monkeypatch, {"approved": True, "severity": "SEV1", "reason": "seen on site", "decided_by": "bob"})

    incident = out["major_incident"]
    assert incident.status == "Declared" and incident.severity == "SEV1"
    assert incident.declared_by == "bob" and incident.declared_at is not None
    assert g.route_after_incident_review({**state, **out}) == "node_generate_communications"

    actions = [e["action"] for e in audit_store.list_audit_events(incident_id=state["incident_id"])]
    assert "declare_major_incident" in actions


@pytest.mark.parametrize("requested", [None, "", "SEV3", "bogus"])
def test_declare_without_valid_severity_defaults_to_sev2(monkeypatch, requested):
    state = _not_candidate_state(monkeypatch)
    out = _review(state, monkeypatch, {"approved": True, "severity": requested})
    assert out["major_incident"].severity == "SEV2"


def test_requested_severity_is_ignored_when_ai_proposed_incident(monkeypatch):
    state = _assessed_state()  # AI a propus SEV2
    out = _review(state, monkeypatch, {"approved": True, "severity": "SEV1"})
    assert out["major_incident"].severity == "SEV2"


def test_dismissed_cluster_is_saved_in_history_but_not_in_memory(isolated, monkeypatch):
    """Realitate: un cluster respins de Assessment trebuie sa apara in istoric; memoria ramane doar pentru decizii umane."""
    monkeypatch.setattr(g, "assess_incident", lambda c, s: _assessment(candidate=False, severity="SEV3"))
    state = {"cluster": _cluster(), "summaries": ["mailbox full", "need larger mailbox", "over quota"]}
    state.update(g.node_assess_incident(state))

    out = g.node_persist_and_learn(state)

    assert out == {"decision_saved": True, "final_status": "DISMISSED_BY_ASSESSMENT"}
    (decision,) = audit_store.list_decisions()
    assert decision["outcome"] == "dismissed"
    assert decision["decided_by"] == "assessment_agent"
    assert decision["severity"] == "SEV3"
    assert decision["incident_id"] == state["incident_id"]
    assert decision["ticket_count"] == 3
    assert decision["communications"] is None
    assert decision["reason"].startswith("Switch down")  # reasoning-ul AI, scurtat
    assert isolated == []  # nimic in memoria Chroma


# ---------- review ----------

def test_approve_declares_with_decider_and_timestamp(monkeypatch):
    state = _assessed_state()
    out = _review(state, monkeypatch, {"approved": True, "decided_by": "alice"})

    incident = out["major_incident"]
    assert incident.status == "Declared"
    assert incident.incident_id == state["incident_id"]
    assert incident.declared_by == "alice"
    assert incident.declared_at is not None
    assert out["user_approved_incident"] is True
    assert g.route_after_incident_review({**state, **out}) == "node_generate_communications"

    actions = [e["action"] for e in audit_store.list_audit_events(incident_id=state["incident_id"])]
    assert "approve_major_incident" in actions


def test_reject_keeps_reason_and_defaults_decider(monkeypatch):
    state = _assessed_state()
    out = _review(state, monkeypatch, {"approved": False, "reason": "single user affected"})

    assert out["major_incident"].status == "Rejected"
    assert out["rejection_reason"] == "single user affected"
    assert out["decided_by"] == "demo_user"
    assert g.route_after_incident_review({**state, **out}) == "node_persist_and_learn"


def test_resume_accepts_json_string(monkeypatch):
    state = _assessed_state()
    out = _review(state, monkeypatch, '{"approved": true, "decided_by": "bob"}')
    assert out["user_approved_incident"] is True
    assert out["decided_by"] == "bob"


# ---------- persist & learn ----------

def test_persist_rejected_saves_decision_and_memory(isolated, monkeypatch):
    state = _assessed_state()
    state.update(_review(state, monkeypatch, {"approved": False, "reason": "false alarm", "decided_by": "alice"}))

    out = g.node_persist_and_learn(state)

    assert out == {"decision_saved": True}
    (decision,) = audit_store.list_decisions()
    assert decision["incident_id"] == state["incident_id"]
    assert decision["outcome"] == "rejected"
    assert decision["reason"] == "false alarm"
    assert decision["decided_by"] == "alice"
    assert decision["communications"] is None
    assert len(isolated) == 1 and isolated[0]["outcome"] == "rejected"


def test_persist_declared_includes_edited_approval_flags(monkeypatch):
    state = _assessed_state()
    state.update(_review(state, monkeypatch, {"approved": True, "decided_by": "alice"}))
    draft = CommunicationDraft(
        incident_id=state["incident_id"], audience="end_users", subject="Network issue", body="We are on it",
        rag_sources=["USER-001"],
    )
    state["communication_drafts"] = {"end_users": draft}
    state["user_approved_communications"] = {"end_users": True}

    g.node_persist_and_learn(state)

    (decision,) = audit_store.list_decisions()
    assert decision["outcome"] == "declared"
    assert decision["communications"]["end_users"]["subject"] == "Network issue"
    assert decision["communications"]["end_users"]["approved"] is True


def test_memory_failure_does_not_stop_flow_but_is_audited(monkeypatch):
    def boom(decision):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(incident_memory, "remember_decision", boom)
    state = _assessed_state()
    state.update(_review(state, monkeypatch, {"approved": False, "reason": "x"}))

    assert g.node_persist_and_learn(state) == {"decision_saved": True}
    assert len(audit_store.list_decisions()) == 1
    actions = [e["action"] for e in audit_store.list_audit_events(incident_id=state["incident_id"])]
    assert "remember_decision_failed" in actions


def test_same_cluster_id_evaluated_twice_keeps_both_decisions(monkeypatch):
    """Regresie: dupa 'incident nou', cluster_id se reia (001) si nu trebuie sa suprascrie decizia veche."""
    for reason in ("first", "second"):
        state = _assessed_state(_cluster("CL-20261006-001"))
        state.update(_review(state, monkeypatch, {"approved": False, "reason": reason}))
        g.node_persist_and_learn(state)

    decisions = audit_store.list_decisions()
    assert len(decisions) == 2
    assert {d["reason"] for d in decisions} == {"first", "second"}
    assert decisions[0]["incident_id"] != decisions[1]["incident_id"]


def test_old_threads_without_incident_id_fall_back(monkeypatch):
    state = _assessed_state()
    state.pop("incident_id")
    out = _review(state, monkeypatch, {"approved": True})
    assert out["major_incident"].incident_id == "MI-CL-20261006-001"


# ---------- structura grafului ----------

def test_graph_structure():
    builder = g.create_workflow()
    assert {
        "node_assess_incident",
        "node_human_review_incident",
        "node_generate_communications",
        "node_human_review_communication",
        "node_persist_and_learn",
    } <= set(builder.nodes)
    # se compileaza fara checkpointer (ca in `langgraph dev`)
    assert builder.compile() is not None


# ---------- comunicari editabile ----------

def _comm_state(monkeypatch):
    state = _assessed_state()
    state.update(_review(state, monkeypatch, {"approved": True, "decided_by": "alice"}))
    state["communication_drafts"] = {
        "end_users": CommunicationDraft(
            incident_id=state["incident_id"], audience="end_users",
            subject="Network issue", body="We are on it", rag_sources=["USER-001"],
        ),
        "management": CommunicationDraft(
            incident_id=state["incident_id"], audience="management",
            subject="SEV2 declared", body="Details for management", rag_sources=["MGMT-001"],
        ),
    }
    return state


def _review_comms(state, monkeypatch, resume):
    monkeypatch.setattr(g, "interrupt", lambda payload: resume)
    return g.node_human_review_communication(state)


def test_edited_communication_is_applied_audited_and_saved(monkeypatch):
    state = _comm_state(monkeypatch)
    out = _review_comms(state, monkeypatch, {
        "approved_users": True,
        "approved_mgmt": True,
        "decided_by": "alice",
        "edits": {"end_users": {"subject": "Network issue (building A)", "body": "We are on it. ETA 30 min."}},
    })

    assert out["communication_drafts"]["end_users"].subject == "Network issue (building A)"
    assert out["communication_drafts"]["management"].subject == "SEV2 declared"  # needitat
    assert out["communication_edits"] == {
        "end_users": {"original_subject": "Network issue", "original_body": "We are on it"}
    }

    events = audit_store.list_audit_events(incident_id=state["incident_id"])
    edit_events = [e for e in events if e["action"] == "edit_communication"]
    assert len(edit_events) == 1 and edit_events[0]["input_ref"] == "end_users"

    state.update(out)
    g.node_persist_and_learn(state)
    comms = audit_store.list_decisions()[0]["communications"]
    assert comms["end_users"]["edited"] is True
    assert comms["end_users"]["body"] == "We are on it. ETA 30 min."
    assert comms["end_users"]["original_body"] == "We are on it"
    assert comms["management"]["edited"] is False
    assert "original_body" not in comms["management"]


def test_unchanged_or_empty_edits_are_ignored(monkeypatch):
    state = _comm_state(monkeypatch)
    out = _review_comms(state, monkeypatch, {
        "approved_users": True,
        "approved_mgmt": True,
        "edits": {
            "end_users": {"subject": "Network issue", "body": "We are on it"},  # identic
            "management": {"subject": "   ", "body": "text"},                    # subiect gol
            "unknown": {"subject": "x", "body": "y"},                            # audienta inexistenta
        },
    })

    assert out["communication_edits"] == {}
    assert out["communication_drafts"]["management"].subject == "SEV2 declared"
    actions = [e["action"] for e in audit_store.list_audit_events(incident_id=state["incident_id"])]
    assert "edit_communication" not in actions


def test_resume_without_edits_still_works(monkeypatch):
    state = _comm_state(monkeypatch)
    out = _review_comms(state, monkeypatch, {"approved_users": True, "approved_mgmt": False})
    assert out["user_approved_communications"] == {"end_users": True, "management": False}
    assert out["communication_edits"] == {}