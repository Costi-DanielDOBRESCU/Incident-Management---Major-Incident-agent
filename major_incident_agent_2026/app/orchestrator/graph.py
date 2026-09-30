"""
app/orchestrator/graph.py
Definirea nodurilor, muchiilor condiționale și construirea grafului LangGraph.
"""

import json
import sqlite3
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from app.orchestrator.state import IncidentState
from app.agents.assessment_agent import assess_incident
from app.agents.communication_agent import generate_communication
from app.models.schemas import (
    IncidentCluster,
    IncidentAssessment,
    MajorIncident,
    CommunicationDraft,
)


def _as(model, value):
    """
    Accepta obiect Pydantic sau dict (cum vine din Studio / langgraph_sdk).
    Intoarce mereu instanta `model` (sau None).
    """
    if value is None or isinstance(value, model):
        return value
    return model(**value)


def _decision(value) -> dict:
    """
    Normalizeaza decizia primita la resume. Accepta dict sau string JSON
    (Studio trimite adesea resume ca text).
    """
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


# ==========================================
# 1. DEFINIRE NODURI (NODE FUNCTIONS)
# ==========================================

def node_assess_incident(state: IncidentState) -> dict:
    """Nod 1: Rulează Assessment Agent pentru clasificare și evaluare."""
    cluster = _as(IncidentCluster, state["cluster"])
    summaries = state.get("summaries", [])

    assessment = assess_incident(cluster, summaries)

    return {
        "cluster": cluster,
        "assessment": assessment,
        "final_status": "ASSESSMENT_COMPLETED",
    }


def node_human_review_incident(state: IncidentState) -> dict:
    """
    Nod 2 (HITL #1): Oprește execuția și așteaptă decizia omului
    privind declararea incidentului major.
    """
    assessment = _as(IncidentAssessment, state.get("assessment"))
    cluster = _as(IncidentCluster, state.get("cluster"))

    # Întrerupem graful și trimitem datele spre UI
    human_decision = interrupt({
        "action": "REVIEW_INCIDENT_DECLARATION",
        "message": "Revizuiește propunerea de Incident Major",
        "assessment": assessment.model_dump() if assessment else None,
    })

    # Decizia primită de la UI prin Command(resume=...)
    human_decision = _decision(human_decision)
    approved = human_decision.get("approved", False)
    reason = human_decision.get("reason", "")

    cluster_id = cluster.cluster_id if cluster else "UNKNOWN"
    severity = assessment.estimated_severity if assessment else "SEV2"

    if approved:
        major_incident = MajorIncident(
            incident_id=f"MI-{cluster_id}",
            cluster_id=cluster_id,
            status="Declared",
            severity=severity,
            declared_by="demo_user",
            root_cause_suspected=assessment.reasoning[:200] if assessment else "",
        )
        return {
            "user_approved_incident": True,
            "major_incident": major_incident,
            "final_status": "INCIDENT_APPROVED",
        }

    major_incident = MajorIncident(
        incident_id=f"MI-{cluster_id}",
        cluster_id=cluster_id,
        status="Rejected",
        severity=severity,
        declared_by="demo_user",
    )
    return {
        "user_approved_incident": False,
        "major_incident": major_incident,
        "rejection_reason": reason,
        "final_status": "REJECTED_BY_USER",
    }


def node_generate_communications(state: IncidentState) -> dict:
    """Nod 3: Generare draft-uri comunicate (end_users & management)."""
    major_incident = _as(MajorIncident, state["major_incident"])
    assessment = _as(IncidentAssessment, state["assessment"])
    cluster = _as(IncidentCluster, state["cluster"])

    drafts = {}
    for audience in ("end_users", "management"):
        drafts[audience] = generate_communication(
            incident=major_incident,
            assessment=assessment,
            cluster=cluster,
            audience=audience,
        )

    return {
        "communication_drafts": drafts,
        "final_status": "DRAFTS_GENERATED",
    }


def node_human_review_communication(state: IncidentState) -> dict:
    """
    Nod 4 (HITL #2): Oprește execuția pentru aprobarea/editarea comunicatelor.
    """
    drafts = {
        k: _as(CommunicationDraft, v)
        for k, v in state.get("communication_drafts", {}).items()
    }

    human_decision = interrupt({
        "action": "REVIEW_COMMUNICATIONS",
        "message": "Revizuiește și aprobă drafturile de comunicare",
        "drafts": {k: v.model_dump() for k, v in drafts.items()},
    })

    human_decision = _decision(human_decision)
    approved_users = human_decision.get("approved_users", False)
    approved_mgmt = human_decision.get("approved_mgmt", False)

    return {
        "user_approved_communications": {
            "end_users": approved_users,
            "management": approved_mgmt,
        },
        "final_status": "COMPLETED_DECLARED",
    }


# ==========================================
# 2. DEFINIRE MUCHII CONDIȚIONALE (ROUTING)
# ==========================================

def route_after_assessment(
    state: IncidentState,
) -> Literal["node_human_review_incident", "__end__"]:
    assessment = _as(IncidentAssessment, state.get("assessment"))
    if assessment and assessment.is_major_incident_candidate:
        return "node_human_review_incident"
    return END


def route_after_incident_review(
    state: IncidentState,
) -> Literal["node_generate_communications", "__end__"]:
    if state.get("user_approved_incident") is True:
        return "node_generate_communications"
    return END


# ==========================================
# 3. CONSTRUIREA GRAFULUI
# ==========================================

def create_workflow():
    builder = StateGraph(IncidentState)

    builder.add_node("node_assess_incident", node_assess_incident)
    builder.add_node("node_human_review_incident", node_human_review_incident)
    builder.add_node("node_generate_communications", node_generate_communications)
    builder.add_node("node_human_review_communication", node_human_review_communication)

    builder.set_entry_point("node_assess_incident")

    # path_map explicit: Studio desenează corect muchiile către END
    builder.add_conditional_edges(
        "node_assess_incident",
        route_after_assessment,
        {
            "node_human_review_incident": "node_human_review_incident",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "node_human_review_incident",
        route_after_incident_review,
        {
            "node_generate_communications": "node_generate_communications",
            END: END,
        },
    )

    builder.add_edge("node_generate_communications", "node_human_review_communication")
    builder.add_edge("node_human_review_communication", END)

    return builder


def compile_graph(db_path: str = "checkpoints.sqlite"):
    """Varianta cu checkpointer SQLite (folosita de Streamlit local)."""
    builder = create_workflow()

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)