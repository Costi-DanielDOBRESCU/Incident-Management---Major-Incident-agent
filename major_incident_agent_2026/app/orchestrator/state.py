"""
app/orchestrator/state.py
Starea globala (IncidentState) transmisa intre nodurile grafului LangGraph.
"""

from typing import Dict, List, Optional, TypedDict

from app.models.schemas import (
    CommunicationDraft,
    IncidentAssessment,
    IncidentCluster,
    MajorIncident,
)


class IncidentState(TypedDict, total=False):
    """
    - cluster: clusterul detectat
    - incident_id: id unic al evaluarii (MI-<cluster>-<sufix>), setat in node_assess_incident
    - summaries: rezumatele tichetelor din cluster
    - assessment: evaluarea Assessment Agent
    - major_incident: obiectul MajorIncident dupa decizia umana (Declared / Rejected)
    - user_approved_incident: decizia umana (HITL #1)
    - rejection_reason: motivul respingerii
    - decided_by: cine a decis (din payload-ul de resume)
    - communication_drafts / user_approved_communications: comunicari si aprobari (HITL #2)
    - communication_edits: {audienta: {original_subject, original_body}} pentru comunicarile editate de om
    - decision_saved: True dupa node_persist_and_learn
    - final_status: ex. COMPLETED_DECLARED, REJECTED_BY_USER
    """
    cluster: IncidentCluster
    incident_id: Optional[str]
    summaries: List[str]
    assessment: Optional[IncidentAssessment]
    major_incident: Optional[MajorIncident]
    user_approved_incident: Optional[bool]
    rejection_reason: Optional[str]
    decided_by: Optional[str]
    communication_drafts: Dict[str, CommunicationDraft]
    communication_edits: Dict[str, dict]
    user_approved_communications: Dict[str, bool]
    decision_saved: Optional[bool]
    final_status: Optional[str]