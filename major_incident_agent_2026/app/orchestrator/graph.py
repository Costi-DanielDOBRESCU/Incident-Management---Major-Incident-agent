"""
app/orchestrator/graph.py
Definirea nodurilor, muchiilor condiționale și construirea grafului LangGraph.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from app.orchestrator.state import IncidentState
from app.agents.assessment_agent import assess_incident
from app.agents.communication_agent import generate_communication
from app.models.schemas import MajorIncident


# ==========================================
# 1. DEFINIRE NODURI (NODE FUNCTIONS)
# ==========================================

def node_assess_incident(state: IncidentState) -> dict:
    """Nod 1: Rulează Assessment Agent pentru clasificare și evaluare."""
    cluster = state["cluster"]
    summaries = state.get("summaries", [])
    
    assessment = assess_incident(cluster, summaries)
    
    return {
        "assessment": assessment,
        "final_status": "ASSESSMENT_COMPLETED"
    }


def node_human_review_incident(state: IncidentState) -> dict:
    """
    Nod 2 (HITL #1): Oprește execuția și așteaptă decizia omului 
    privind declararea incidentului major.
    """
    assessment = state.get("assessment")
    
    # Întrerupem graful și trimitem datele spre interfața utilizator (Streamlit/UI)
    human_decision = interrupt({
        "action": "REVIEW_INCIDENT_DECLARATION",
        "message": "Revizuiește propunerea de Incident Major",
        "assessment": assessment.model_dump() if assessment else None
    })
    
    # Decizia primită de la UI în timpul resume-ului
    approved = human_decision.get("approved", False)
    reason = human_decision.get("reason", "")
    
    if approved:
        major_incident = MajorIncident(
            title=f"Incident Major: {assessment.primary_issue if assessment else 'Nespecificat'}",
            description=assessment.rationale if assessment else "",
            severity=assessment.severity if assessment else "SEV2",
            affected_service=assessment.affected_service if assessment else "Unknown",
            cluster_id=state["cluster"].cluster_id
        )
        return {
            "user_approved_incident": True,
            "major_incident": major_incident,
            "final_status": "INCIDENT_APPROVED"
        }
    else:
        return {
            "user_approved_incident": False,
            "rejection_reason": reason,
            "final_status": "REJECTED_BY_USER"
        }


def node_generate_communications(state: IncidentState) -> dict:
    """Nod 3: Generare draft-uri comunicate (end_users & management)."""
    major_incident = state["major_incident"]
    assessment = state["assessment"]
    
    # Generăm draft pentru End Users
    draft_end_users = generate_communication(
        incident=major_incident,
        assessment=assessment,
        audience="end_users"
    )
    
    # Generăm draft pentru Management
    draft_management = generate_communication(
        incident=major_incident,
        assessment=assessment,
        audience="management"
    )
    
    return {
        "communication_drafts": {
            "end_users": draft_end_users,
            "management": draft_management
        },
        "final_status": "DRAFTS_GENERATED"
    }


def node_human_review_communication(state: IncidentState) -> dict:
    """
    Nod 4 (HITL #2): Oprește execuția pentru aprobarea/editarea comunicatelor.
    """
    drafts = state.get("communication_drafts", {})
    
    human_decision = interrupt({
        "action": "REVIEW_COMMUNICATIONS",
        "message": "Revizuiește și aprobă drafturile de comunicare",
        "drafts": {k: v.model_dump() for k, v in drafts.items()}
    })
    
    approved_users = human_decision.get("approved_users", False)
    approved_mgmt = human_decision.get("approved_mgmt", False)
    
    return {
        "user_approved_communications": {
            "end_users": approved_users,
            "management": approved_mgmt
        },
        "final_status": "COMPLETED_DECLARED"
    }


# ==========================================
# 2. DEFINIRE MUCHII CONDIȚIONALE (ROUTING)
# ==========================================

def route_after_assessment(state: IncidentState) -> Literal["node_human_review_incident", "__end__"]:
    """Evaluează dacă incidentul este candidat de Incident Major."""
    assessment = state.get("assessment")
    if assessment and assessment.is_major_incident_candidate:
        return "node_human_review_incident"
    return END


def route_after_incident_review(state: IncidentState) -> Literal["node_generate_communications", "__end__"]:
    """Verifică dacă utilizatorul a aprobat declararea incidentului."""
    if state.get("user_approved_incident") is True:
        return "node_generate_communications"
    return END


# ==========================================
# 3. CONSTRUIREA GRAFULUI
# ==========================================

def create_workflow():
    builder = StateGraph(IncidentState)
    
    # Adăugare Noduri
    builder.add_node("node_assess_incident", node_assess_incident)
    builder.add_node("node_human_review_incident", node_human_review_incident)
    builder.add_node("node_generate_communications", node_generate_communications)
    builder.add_node("node_human_review_communication", node_human_review_communication)
    
    # Setare punct de intrare
    builder.set_entry_point("node_assess_incident")
    
    # Legături / Control Flow
    builder.add_conditional_edges(
        "node_assess_incident",
        route_after_assessment
    )
    
    builder.add_conditional_edges(
        "node_human_review_incident",
        route_after_incident_review
    )
    
    builder.add_edge("node_generate_communications", "node_human_review_communication")
    builder.add_edge("node_human_review_communication", END)
    
    return builder


def compile_graph(db_path: str = "checkpoints.sqlite"):
    """
    Compilează graful atașând un SqliteSaver pentru persistența stării (Checkpointing).
    """
    builder = create_workflow()
    
    # Configurare persistență SQLite pentru pauză/reluare (HITL)
    conn = SqliteSaver.from_conn_string(db_path)
    graph = builder.compile(checkpointer=conn)
    
    return graph