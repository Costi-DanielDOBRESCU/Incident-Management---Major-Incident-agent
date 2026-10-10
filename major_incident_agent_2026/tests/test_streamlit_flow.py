"""
Teste de flux pentru UI (Streamlit AppTest): navigare intr-un click, butoane corecte la fiecare pas,
refacere dupa erori de LLM, protectii in Sumar. Backend-ul LangGraph si detectia sunt simulate;
bazele de date sunt temporare (nu ating tichetele sau istoricul real).
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import httpx  # noqa: E402
import langgraph_sdk  # noqa: E402
import streamlit as st  # noqa: E402

import app.detection.pipeline as pipeline  # noqa: E402
from app.execution import audit_store  # noqa: E402
from app.ingestion import ticket_store  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app" / "ui" / "streamlit_app.py"

ASSESS = {"cluster_id": "CL-1", "is_major_incident_candidate": True, "confidence": 0.93, "estimated_severity": "SEV2",
          "affected_service": "VPN Gateway", "reasoning": "shared cause", "rag_sources": ["RB-VPN-002"],
          "recommended_action": "propose_major_incident"}
DRAFTS = {
    "end_users": {"incident_id": "MI-1", "audience": "end_users", "subject": "VPN issue", "body": "We are on it", "rag_sources": ["USER-001"]},
    "management": {"incident_id": "MI-1", "audience": "management", "subject": "SEV2 declared", "body": "Mgmt text", "rag_sources": ["MGMT-001"]},
}
START = "Next: Evaluare AI (Pornire Graf LangGraph)"


class Backend:
    """Simuleaza serverul LangGraph: stare per thread, erori controlabile, reluare de la checkpoint."""

    def __init__(self):
        self.threads, self.calls = {}, []
        self.fail_assess = self.fail_comms = False
        self.candidate = True

    def _assess(self, th, inp):
        if self.fail_assess:
            raise RuntimeError("429 rate limit")
        a = dict(ASSESS, is_major_incident_candidate=self.candidate, estimated_severity="SEV2" if self.candidate else "SEV3")
        th["values"].update(cluster=inp["cluster"], summaries=inp["summaries"], assessment=a, incident_id="MI-1")
        th["next"] = ["node_human_review_incident"]

    def _comms(self, th):
        if self.fail_comms:
            raise RuntimeError("429 rate limit")
        th["values"]["communication_drafts"] = copy.deepcopy(DRAFTS)
        th["next"] = ["node_human_review_communication"]

    def client(self):
        b = self

        class Threads:
            def create(self):
                tid = f"t{len(b.threads) + 1}"
                b.threads[tid] = {"values": {}, "next": [], "input": None}
                return {"thread_id": tid}

            def get_state(self, tid):
                return copy.deepcopy(b.threads[tid])

        class Runs:
            def wait(self, tid, graph, input=None, command=None):
                th = b.threads[tid]
                if command is None and input is not None:
                    b.calls.append(("start", None)); th["input"] = input; th["next"] = ["node_assess_incident"]
                    b._assess(th, input)
                    return
                if command is None:  # reluare de la ultimul checkpoint
                    b.calls.append(("retry", tuple(th["next"])))
                    if th["next"] == ["node_assess_incident"]:
                        b._assess(th, th["input"])
                    elif th["next"] == ["node_generate_communications"]:
                        b._comms(th)
                    return
                r, v = command["resume"], th["values"]
                b.calls.append(("resume", r))
                if th["next"] == ["node_human_review_incident"]:
                    v["user_approved_incident"] = bool(r["approved"])
                    if r["approved"]:
                        th["next"] = ["node_generate_communications"]; b._comms(th)
                    else:
                        th["next"] = []; v.update(decision_saved=True, final_status="REJECTED_BY_USER")
                elif th["next"] == ["node_human_review_communication"]:
                    for aud, e in (r.get("edits") or {}).items():
                        v["communication_drafts"][aud].update(e)
                    v["user_approved_communications"] = {"end_users": r["approved_users"], "management": r["approved_mgmt"]}
                    th["next"] = []; v.update(decision_saved=True, final_status="COMPLETED_DECLARED")

        class C:
            threads, runs = Threads(), Runs()

        return C()


def _issue(i):
    return {"key": f"INC-{i}", "fields": {
        "summary": f"VPN not connecting {i}", "description": "cannot connect", "created": f"2026-10-09T09:{10 + i:02d}:00.000+0000",
        "priority": {"name": "P2 - High"}, "status": {"name": "Open"}, "components": [{"name": "VPN Gateway"}],
        "reporter": {"name": "user_101"}, "labels": ["network"], "location": "RO-Timisoara"}}


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    """Fabrica de aplicatii: DB-uri temporare, fara retea, backend LangGraph simulat."""
    monkeypatch.setattr(ticket_store, "DB_PATH", tmp_path / "tickets.db")
    monkeypatch.setattr(audit_store, "DB_PATH", tmp_path / "audit.db")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: type("R", (), {"json": lambda self: {}})())
    monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"status_code": 200})())

    def build(backend=None, clusters=((0, 1, 2),), n_tickets=6, **state):
        backend = backend or Backend()
        ticket_store.reset_db(); ticket_store.init_db()
        ticket_store.upsert_tickets([_issue(i) for i in range(1, n_tickets + 1)])

        def fake_detect(tickets, embed=None):
            n = len(tickets)
            m = np.full((n, n), 0.3); np.fill_diagonal(m, 1.0)
            for group in clusters:
                for a in group:
                    for c in group:
                        m[a][c] = 0.85
            return [list(g) for g in clusters], m

        st.cache_resource.clear(); st.cache_data.clear()
        monkeypatch.setattr(langgraph_sdk, "get_sync_client", lambda **kw: backend.client())
        monkeypatch.setattr(pipeline, "detect_clusters_with_matrix", fake_detect)
        at = AppTest.from_file(str(APP), default_timeout=60)
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, at.exception
        return at, backend

    return build


def labels(at):
    return [b.label for b in at.button]


def click(at, label):
    button = next((b for b in at.button if b.label == label), None)
    assert button is not None, f"butonul {label!r} lipseste; butoane: {labels(at)}"
    at = button.click().run()
    assert not at.exception, at.exception
    return at


def step(at):
    return at.session_state["wizard_step"]


def to_decision(at):
    for label in ("Next: Detecție & clustering", START, "Next: Aprobare umană"):
        at = click(at, label)
    return at


# ---------- navigare intr-un click ----------

def test_declared_flow_moves_forward_with_one_click_per_step(make_app):
    at, backend = make_app()
    assert labels(at) == ["Next: Detecție & clustering"]

    at = click(at, "Next: Detecție & clustering")           # detectia ruleaza in acelasi click
    assert step(at) == 1 and START in labels(at)
    at = click(at, START)
    assert step(at) == 2 and "Next: Aprobare umană" in labels(at)
    at = click(at, "Next: Aprobare umană")
    assert step(at) == 3 and {"Aprobă Major Incident", "Respinge", "Back"} <= set(labels(at))

    at = click(at, "Aprobă Major Incident")                 # direct la comunicari, fara "Next: Comunicări"
    assert step(at) == 4
    assert "Next: Sumar" not in labels(at), "nu se poate sari peste aprobarea comunicarilor"
    assert labels(at) == ["Back", "Aprobă și Finalizează Comunicările"]

    at = click(at, "Aprobă și Finalizează Comunicările")    # direct la sumar
    assert step(at) == 5 and labels(at) == ["Înapoi la clustere", "Începe un incident nou"]
    assert backend.calls[-1][0] == "resume"


def test_edited_communications_reach_the_graph_and_only_for_the_edited_audience(make_app):
    at, backend = make_app()
    at = to_decision(at)
    at = click(at, "Aprobă Major Incident")
    assert all(not w.disabled for w in at.text_area) and all(not w.disabled for w in at.text_input)

    at.text_input(key="comm_subject_t1_end_users").set_value("VPN issue (editat)").run()
    at.text_area(key="comm_body_t1_end_users").set_value("Text editat").run()
    click(at, "Aprobă și Finalizează Comunicările")

    resume = [c[1] for c in backend.calls if c[0] == "resume"][-1]
    assert resume["edits"] == {"end_users": {"subject": "VPN issue (editat)", "body": "Text editat"}}


def test_edited_text_survives_navigating_away_and_back(make_app):
    at, _ = make_app()
    at = to_decision(at)
    at = click(at, "Aprobă Major Incident")
    at.text_area(key="comm_body_t1_management").set_value("Text management editat").run()

    at = click(at, "Back")
    assert step(at) == 3
    at = click(at, "Next: Comunicări")
    assert at.text_area(key="comm_body_t1_management").value == "Text management editat"


def test_reject_goes_straight_to_summary_and_sends_the_reason(make_app):
    at, backend = make_app()
    at = to_decision(at)
    at.text_input(key="decision_reason").set_value("doar o licenta").run()
    at = click(at, "Respinge")

    assert step(at) == 5
    assert [c[1] for c in backend.calls if c[0] == "resume"][-1]["reason"] == "doar o licenta"


@pytest.mark.parametrize("action, expected_step, severity", [
    ("Confirm: nu e incident", 5, None),
    ("Declar incident major", 4, "SEV1"),
])
def test_cluster_not_proposed_by_ai_needs_a_human_decision(make_app, action, expected_step, severity):
    backend = Backend(); backend.candidate = False
    at, _ = make_app(backend)
    at = to_decision(at)

    assert "Confirm: nu e incident" in labels(at) and "Declar incident major" in labels(at)
    assert "Aprobă Major Incident" not in labels(at)
    assert at.selectbox(key="declare_severity").options == ["SEV2", "SEV1"]
    if severity:
        at.selectbox(key="declare_severity").set_value(severity).run()
    at = click(at, action)

    assert step(at) == expected_step
    assert [c[1] for c in backend.calls if c[0] == "resume"][-1].get("severity") == severity


# ---------- refacere dupa erori ----------

def test_assessment_failure_can_be_retried(make_app):
    backend = Backend(); backend.fail_assess = True
    at, _ = make_app(backend)
    at = click(at, "Next: Detecție & clustering")
    at = click(at, START)
    assert at.error and step(at) == 1

    at.session_state["wizard_step"] = 2
    at = at.run()
    assert labels(at) == ["Back", "Reia evaluarea"]

    backend.fail_assess = False
    at = click(at, "Reia evaluarea")
    assert "Next: Aprobare umană" in labels(at)


def test_communications_failure_keeps_the_decision_and_can_be_retried(make_app):
    backend = Backend(); backend.fail_comms = True
    at, _ = make_app(backend)
    at = to_decision(at)
    at = click(at, "Aprobă Major Incident")
    assert at.error and backend.threads["t1"]["values"]["user_approved_incident"] is True

    at.session_state["wizard_step"] = 3
    at = click(at.run(), "Next: Comunicări")
    assert labels(at) == ["Back", "Reia generarea comunicărilor"]

    backend.fail_comms = False
    at = click(at, "Reia generarea comunicărilor")
    assert labels(at) == ["Back", "Aprobă și Finalizează Comunicările"]
    assert all(not w.disabled for w in at.text_area)


def test_decision_step_offers_retry_when_graph_is_not_waiting_for_a_decision(make_app):
    at, backend = make_app()
    at = to_decision(at)
    backend.threads["t1"]["next"] = ["node_assess_incident"]
    at = at.run()

    assert "Reia execuția" in labels(at) and "Aprobă Major Incident" not in labels(at)


# ---------- sumar si clustere ----------

def test_summary_protects_clusters_that_are_not_finished(make_app):
    at, _ = make_app(clusters=((0, 1, 2), (3, 4, 5)))
    at = to_decision(at)
    at = click(at, "Respinge")                                # primul cluster terminat, al doilea neevaluat
    assert step(at) == 5 and at.warning

    new_incident = next(b for b in at.button if b.label == "Începe un incident nou")
    assert new_incident.disabled, "nu se poate renunta la clustere fara confirmare"
    at.checkbox[0].check().run()
    assert not next(b for b in at.button if b.label == "Începe un incident nou").disabled

    at = click(at, "Înapoi la clustere")
    assert step(at) == 1 and at.radio[0].value == "CL-20261009-002"   # cel neevaluat e preselectat
    assert START in labels(at)


def test_summary_without_unfinished_clusters_has_no_warning(make_app):
    at, _ = make_app()
    at = to_decision(at)
    at = click(at, "Respinge")

    assert not at.warning and not at.checkbox
    assert not next(b for b in at.button if b.label == "Începe un incident nou").disabled


def test_summary_warns_when_a_declared_incident_has_unapproved_communications(make_app):
    at, _ = make_app()
    at = to_decision(at)
    at = click(at, "Aprobă Major Incident")
    at.session_state["wizard_step"] = 5
    at = at.run()

    assert "Continuă: aprobă comunicările" in labels(at)
    at = click(at, "Continuă: aprobă comunicările")
    assert step(at) == 4


def test_tickets_step_keeps_evaluated_clusters_unless_detection_is_rerun(make_app):
    at, _ = make_app()
    at = to_decision(at)
    at.session_state["wizard_step"] = 0
    at = at.run()

    assert "Next: Detecție & clustering" not in labels(at)
    assert "Reia detecția cu tichetele curente" in labels(at)
    at = click(at, next(l for l in labels(at) if l.startswith("Înapoi la clustere")))
    assert step(at) == 1 and at.session_state["cluster_threads"]

    at.session_state["wizard_step"] = 0
    at = click(at.run(), "Reia detecția cu tichetele curente")
    assert step(at) == 1 and at.session_state["cluster_threads"] == {}


# ---------- antet si istoric ----------

def _header(at):
    return next(m.value for m in at.markdown if 'class="itsm-header-banner"' in m.value and "Major Incident Management" in m.value)


def test_server_indicator_reflects_the_real_connection(make_app, monkeypatch):
    at, _ = make_app()
    assert "LangGraph conectat" in _header(at) and not at.error

    def down(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", down)
    at, _ = make_app()
    assert "LangGraph indisponibil" in _header(at) and "dot off" in _header(at)
    assert any("langgraph dev" in e.value for e in at.error)


def test_history_page_counts_human_decisions_against_ai_proposals(make_app):
    at, _ = make_app()
    a = dict(ASSESS)
    for i, (outcome, ai_candidate) in enumerate([("declared", True), ("rejected", True), ("rejected", False), ("declared", False)]):
        audit_store.save_decision({
            "incident_id": f"MI-{i}", "cluster_id": "c", "service": "VPN Gateway", "severity": "SEV2", "outcome": outcome,
            "decided_by": "alice", "ticket_ids": ["T-1"], "ticket_count": 1, "summaries": ["s"],
            "assessment": dict(a, is_major_incident_candidate=ai_candidate)})

    at.sidebar.radio(key="nav_view").set_value("Istoric incidente").run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["4", "2", "2", "2"]