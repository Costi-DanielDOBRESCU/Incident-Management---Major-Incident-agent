"""
app/orchestrator/state.py
Definirea stării globale (IncidentState) transmise între nodurile grafului LangGraph.
"""

from typing import TypedDict, List, Optional, Dict
from app.models.schemas import (
    IncidentCluster,
    IncidentAssessment,
    MajorIncident,
    CommunicationDraft
)


class IncidentState(TypedDict, total=False):
    """
    Starea care circulă prin nodurile grafului LangGraph.
    
    Câmpuri:
    - cluster: Clusterul detectat de bilete/incidente (obiect IncidentCluster)
    - summaries: Lista rezumatelor/titlurilor text ale biletelor din cluster
    - assessment: Evaluarea realizată de Assessment Agent (dacă e incident major candidate, severitate, motivare)
    - major_incident: Obiectul de tip MajorIncident format după confirmarea/evaluarea incidentului
    - user_approved_incident: Flag de confirmare umane (Human-In-The-Loop) pentru declararea incidentului major
    - rejection_reason: Motivul respingerii în cazul în care utilizatorul nu aprobă incidentul major
    - communication_drafts: Dictionar cu draft-urile generate per audiență {"end_users": Draft, "management": Draft}
    - user_approved_communications: Flag-uri de aprobare umane per comunicat {"end_users": True/False, "management": True/False}
    - final_status: Starea finală a fluxului (ex: "COMPLETED_DECLARED", "REJECTED_BY_USER", "NOT_MAJOR_INCIDENT")
    """
    cluster: IncidentCluster
    summaries: List[str]
    assessment: Optional[IncidentAssessment]
    major_incident: Optional[MajorIncident]
    user_approved_incident: Optional[bool]
    rejection_reason: Optional[str]
    communication_drafts: Dict[str, CommunicationDraft]
    user_approved_communications: Dict[str, bool]
    final_status: Optional[str]