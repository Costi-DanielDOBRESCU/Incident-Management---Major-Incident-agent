"""
app/orchestrator/graph.py
Noduri, muchii conditionale si constructia grafului LangGraph.

Persistare + invatare:
  - audit (SQLite) in noduri, DUPA interrupt() (nodul se reia la resume, deci inainte de interrupt s-ar dubla);
  - node_execute_actions: Execution Layer (notificari simulate, status + link tichete), doar dupa aprobarea omului;
  - node_persist_and_learn: salveaza decizia umana in SQLite, apoi (best effort) in memoria Chroma.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.assessment_agent import assess_incident
from app.agents.communication_agent import generate_communication
from app.agents.llm_client import last_llm_label
from app.execution import audit_store
from app.models.schemas import (
    CommunicationDraft,
    IncidentAssessment,
    IncidentCluster,
    MajorIncident,
)
from app.orchestrator.state import IncidentState

DEFAULT_USER = "demo_user"


def _as(model, value):
    """Accepta obiect Pydantic sau dict (Studio / langgraph_sdk). Intoarce instanta `model` (sau None)."""
    if value is None or isinstance(value, model):
        return value
    return model(**value)


def _decision(value) -> dict:
    """Normalizeaza decizia primita la resume (dict sau string JSON)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"approved": value.strip().lower() in ("true", "yes", "da", "approve", "approved")}
    return {}


def _new_incident_id(cluster_id: str) -> str:
    """
    Id unic per evaluare. cluster_id se renumeroteaza de la 001 la fiecare detectie,
    deci doar MI-<cluster_id> ar suprascrie in audit.db decizii mai vechi.
    """
    return f"MI-{cluster_id}-{uuid.uuid4().hex[:6]}"


def _incident_id(state: IncidentState, cluster_id: str) -> str:
    """Id-ul din state (setat in node_assess_incident); fallback pentru thread-uri create inainte de aceasta schimbare."""
    return state.get("incident_id") or f"MI-{cluster_id}"


def _audit(actor: str, action: str, **kwargs) -> None:
    """Audit best effort: o eroare de scriere nu trebuie sa opreasca fluxul."""
    try:
        audit_store.log_audit_event(actor, action, **kwargs)
    except Exception:  # noqa: BLE001
        pass


# ==========================================
# 1. NODURI
# ==========================================
def node_assess_incident(state: IncidentState) -> dict:
    """Nod 1: Assessment Agent (RAG + LLM)."""
    cluster = _as(IncidentCluster, state["cluster"])
    summaries = state.get("summaries", [])

    assessment = assess_incident(cluster, summaries)
    incident_id = _new_incident_id(cluster.cluster_id)

    _audit(
        "assessment_agent",
        "propose_major_incident" if assessment.is_major_incident_candidate else "assess_not_candidate",
        incident_id=incident_id,
        input_ref=cluster.cluster_id,
        output_ref=f"{assessment.estimated_severity}/{assessment.recommended_action}",
        model=last_llm_label(),
        rag_sources=list(assessment.rag_sources),
        based_on={
            "ticket_count": cluster.ticket_count,
            "service": assessment.affected_service,
            "confidence": assessment.confidence,
        },
    )

    return {
        "cluster": cluster,
        "assessment": assessment,
        "incident_id": incident_id,
        "final_status": "ASSESSMENT_COMPLETED",
    }


def node_human_review_incident(state: IncidentState) -> dict:
    """
    Nod 2 (HITL #1): decizia omului asupra declararii. Se ruleaza SI cand Assessment nu a propus incident:
    omul confirma ca nu e incident (approved=False) sau il declara el (approved=True, cu severitate aleasa).
    """
    assessment = _as(IncidentAssessment, state.get("assessment"))
    cluster = _as(IncidentCluster, state.get("cluster"))
    ai_candidate = bool(assessment and assessment.is_major_incident_candidate)

    raw = interrupt({
        "action": "REVIEW_INCIDENT_DECLARATION",
        "message": "Revizuieste propunerea de Incident Major" if ai_candidate
        else "Assessment nu a propus incident major: confirma sau declara tu incidentul",
        "ai_candidate": ai_candidate,
        "assessment": assessment.model_dump() if assessment else None,
    })

    # --- tot ce urmeaza ruleaza dupa resume ---
    human = _decision(raw)
    approved = bool(human.get("approved", False))
    reason = (human.get("reason") or "").strip()
    decided_by = human.get("decided_by") or DEFAULT_USER

    cluster_id = cluster.cluster_id if cluster else "UNKNOWN"
    incident_id = _incident_id(state, cluster_id)
    if approved and not ai_candidate:
        # AI nu a propus incident: severitatea o alege omul (SEV1/SEV2); orice altceva -> SEV2
        requested = str(human.get("severity") or "").upper()
        severity = requested if requested in ("SEV1", "SEV2") else "SEV2"
    else:
        severity = assessment.estimated_severity if assessment else "SEV2"

    if ai_candidate:
        action = "approve_major_incident" if approved else "reject_major_incident"
    else:
        action = "declare_major_incident" if approved else "confirm_not_incident"

    _audit(
        decided_by,
        action,
        incident_id=incident_id,
        input_ref=cluster_id,
        output_ref="Declared" if approved else "Rejected",
        rag_sources=list(assessment.rag_sources) if assessment else [],
        based_on={"severity": severity, "reason": reason or None, "ai_candidate": ai_candidate},
    )

    if approved:
        major_incident = MajorIncident(
            incident_id=incident_id,
            cluster_id=cluster_id,
            status="Declared",
            severity=severity,
            declared_by=decided_by,
            declared_at=datetime.now(timezone.utc),
            root_cause_suspected=assessment.reasoning[:200] if assessment else "",
        )
        return {
            "user_approved_incident": True,
            "major_incident": major_incident,
            "rejection_reason": None,
            "decided_by": decided_by,
            "final_status": "INCIDENT_APPROVED",
        }

    major_incident = MajorIncident(
        incident_id=incident_id,
        cluster_id=cluster_id,
        status="Rejected",
        severity=severity,
        declared_by=decided_by,
    )
    return {
        "user_approved_incident": False,
        "major_incident": major_incident,
        "rejection_reason": reason,
        "decided_by": decided_by,
        "final_status": "REJECTED_BY_USER",
    }


def node_generate_communications(state: IncidentState) -> dict:
    """Nod 3: drafturi de comunicare (end_users & management)."""
    major_incident = _as(MajorIncident, state["major_incident"])
    assessment = _as(IncidentAssessment, state["assessment"])
    cluster = _as(IncidentCluster, state["cluster"])

    drafts = {}
    for audience in ("end_users", "management"):
        draft = generate_communication(
            incident=major_incident,
            assessment=assessment,
            cluster=cluster,
            audience=audience,
        )
        drafts[audience] = draft
        _audit(
            "communication_agent",
            "draft_communication",
            incident_id=major_incident.incident_id,
            input_ref=audience,
            output_ref=draft.subject,
            model=last_llm_label(),
            rag_sources=list(draft.rag_sources),
        )

    return {"communication_drafts": drafts, "final_status": "DRAFTS_GENERATED"}


def node_human_review_communication(state: IncidentState) -> dict:
    """Nod 4 (HITL #2): aprobarea comunicatelor."""
    drafts = {
        k: _as(CommunicationDraft, v)
        for k, v in state.get("communication_drafts", {}).items()
    }
    incident_id = _as(MajorIncident, state["major_incident"]).incident_id

    raw = interrupt({
        "action": "REVIEW_COMMUNICATIONS",
        "message": "Revizuieste si aproba drafturile de comunicare",
        "drafts": {k: v.model_dump() for k, v in drafts.items()},
    })

    human = _decision(raw)
    approved_users = bool(human.get("approved_users", False))
    approved_mgmt = bool(human.get("approved_mgmt", False))
    decided_by = human.get("decided_by") or state.get("decided_by") or DEFAULT_USER

    # Editarile omului (doar cele care chiar schimba textul). Se pastreaza originalul pentru audit.
    edits_meta: dict[str, dict] = {}
    for audience, edit in (human.get("edits") or {}).items():
        draft = drafts.get(audience)
        if draft is None or not isinstance(edit, dict):
            continue
        subject = (edit.get("subject") or "").strip()
        body = (edit.get("body") or "").strip()
        if not subject or not body:
            continue  # nu se accepta comunicari goale
        if subject == draft.subject and body == draft.body:
            continue
        edits_meta[audience] = {"original_subject": draft.subject, "original_body": draft.body}
        drafts[audience] = draft.model_copy(update={"subject": subject, "body": body})
        _audit(
            decided_by,
            "edit_communication",
            incident_id=incident_id,
            input_ref=audience,
            output_ref=subject,
            based_on={
                "subject_changed": subject != draft.subject,
                "body_changed": body != draft.body,
            },
        )

    for audience, ok in (("end_users", approved_users), ("management", approved_mgmt)):
        _audit(
            decided_by,
            "approve_communication" if ok else "reject_communication",
            incident_id=incident_id,
            input_ref=audience,
        )

    return {
        "communication_drafts": drafts,
        "communication_edits": edits_meta,
        "user_approved_communications": {"end_users": approved_users, "management": approved_mgmt},
        "decided_by": decided_by,
        "final_status": "COMPLETED_DECLARED",
    }


def _build_decision(state: IncidentState) -> dict:
    """
    Decizia ca dict pentru audit_store / incident_memory. Doar date deja validate (Pydantic).
    outcome: declared | rejected (decizie umana) sau dismissed (Assessment nu a propus incident; fara decizie umana).
    """
    cluster = _as(IncidentCluster, state["cluster"])
    assessment = _as(IncidentAssessment, state.get("assessment"))
    incident = _as(MajorIncident, state.get("major_incident"))
    now = datetime.now(timezone.utc)

    if incident is None:
        # Nu a existat review uman: Assessment a spus ca nu e candidat. Se pastreaza in istoric, nu in memoria de decizii umane.
        return {
            "incident_id": state.get("incident_id") or f"MI-{cluster.cluster_id}",
            "cluster_id": cluster.cluster_id,
            "service": assessment.affected_service if assessment else cluster.service_guess,
            "severity": assessment.estimated_severity if assessment else None,
            "outcome": "dismissed",
            "reason": (assessment.reasoning[:300] if assessment else None),
            "decided_by": "assessment_agent",
            "decided_at": now.isoformat(),
            "ticket_ids": list(cluster.ticket_ids),
            "ticket_count": cluster.ticket_count,
            "summaries": list(state.get("summaries") or []),
            "assessment": assessment.model_dump() if assessment else None,
            "communications": None,
        }

    approved = state.get("user_approved_incident") is True

    communications = None
    drafts = state.get("communication_drafts") or {}
    if approved and drafts:
        flags = state.get("user_approved_communications") or {}
        edits = state.get("communication_edits") or {}
        communications = {}
        for audience, d in drafts.items():
            d = _as(CommunicationDraft, d)
            entry = {
                "subject": d.subject,  # textul FINAL (dupa editarea omului, daca a existat)
                "body": d.body,
                "rag_sources": list(d.rag_sources),
                "approved": bool(flags.get(audience, False)),
                "edited": audience in edits,
            }
            if audience in edits:
                entry["original_subject"] = edits[audience]["original_subject"]
                entry["original_body"] = edits[audience]["original_body"]
            communications[audience] = entry

    return {
        "incident_id": incident.incident_id,
        "cluster_id": cluster.cluster_id,
        "service": assessment.affected_service if assessment else cluster.service_guess,
        "severity": incident.severity,
        "outcome": "declared" if approved else "rejected",
        "reason": state.get("rejection_reason") or None,
        "decided_by": state.get("decided_by") or DEFAULT_USER,
        "decided_at": (incident.declared_at or now).isoformat(),
        "ticket_ids": list(cluster.ticket_ids),
        "ticket_count": cluster.ticket_count,
        "summaries": list(state.get("summaries") or []),
        "assessment": assessment.model_dump() if assessment else None,
        "communications": communications,
    }


def node_execute_actions(state: IncidentState) -> dict:
    """
    Nod 4b: Execution Layer (determinist). Ruleaza DOAR dupa ambele puncte HITL.
    Executa doar ce a aprobat omul (vezi app/execution/tools.run_execution).
    Erorile se auditeaza, dar nu opresc fluxul. NU modifica final_status (il citeste UI-ul).
    """
    decision = _build_decision(state)
    incident_id = decision["incident_id"]
    try:
        from app.execution.tools import run_execution

        result = run_execution(decision)
        _audit(
            "execution_layer",
            "execute_actions",
            incident_id=incident_id,
            output_ref="executed" if result["executed"] else "skipped",
            based_on=result,
        )
    except Exception as exc:  # noqa: BLE001
        result = {"executed": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        _audit("execution_layer", "execute_actions_failed", incident_id=incident_id, output_ref=result["error"])

    return {"execution_result": result}


def node_persist_and_learn(state: IncidentState) -> dict:
    """
    Nod 5: salveaza decizia umana (SQLite), apoi o adauga in memoria de incidente (Chroma).
    Esecul memoriei se auditeaza, dar nu opreste fluxul.
    """
    decision = _build_decision(state)
    incident_id = decision["incident_id"]

    audit_store.save_decision(decision)
    _audit(
        "persistence_layer",
        "save_decision",
        incident_id=incident_id,
        output_ref=decision["outcome"],
    )

    if decision["outcome"] == "dismissed":
        # memoria de incidente contine doar decizii UMANE (declared/rejected)
        return {"decision_saved": True, "final_status": "DISMISSED_BY_ASSESSMENT"}

    try:
        from app.execution.incident_memory import remember_decision

        doc_id = remember_decision(decision)
        _audit("persistence_layer", "remember_decision", incident_id=incident_id, output_ref=doc_id)
    except Exception as exc:  # noqa: BLE001
        _audit(
            "persistence_layer",
            "remember_decision_failed",
            incident_id=incident_id,
            output_ref=f"{type(exc).__name__}: {exc}"[:300],
        )

    return {"decision_saved": True}


# ==========================================
# 2. ROUTARE
# ==========================================
def route_after_incident_review(
    state: IncidentState,
) -> Literal["node_generate_communications", "node_persist_and_learn"]:
    if state.get("user_approved_incident") is True:
        return "node_generate_communications"
    return "node_persist_and_learn"


# ==========================================
# 3. GRAF
# ==========================================
def create_workflow():
    builder = StateGraph(IncidentState)

    builder.add_node("node_assess_incident", node_assess_incident)
    builder.add_node("node_human_review_incident", node_human_review_incident)
    builder.add_node("node_generate_communications", node_generate_communications)
    builder.add_node("node_human_review_communication", node_human_review_communication)
    builder.add_node("node_execute_actions", node_execute_actions)
    builder.add_node("node_persist_and_learn", node_persist_and_learn)

    builder.set_entry_point("node_assess_incident")

    # Orice cluster evaluat trece prin om, chiar daca Assessment nu l-a propus ca incident
    builder.add_edge("node_assess_incident", "node_human_review_incident")
    builder.add_conditional_edges(
        "node_human_review_incident",
        route_after_incident_review,
        {
            "node_generate_communications": "node_generate_communications",
            "node_persist_and_learn": "node_persist_and_learn",
        },
    )
    builder.add_edge("node_generate_communications", "node_human_review_communication")
    builder.add_edge("node_human_review_communication", "node_execute_actions")
    builder.add_edge("node_execute_actions", "node_persist_and_learn")
    builder.add_edge("node_persist_and_learn", END)

    return builder


def compile_graph(db_path: str = "checkpoints.sqlite"):
    """Varianta cu checkpointer SQLite (folosita de Streamlit local)."""
    builder = create_workflow()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))