"""
Modele Pydantic - contractul de date al intregului sistem MIA.

Aceste modele sunt granita intre:
  - reasoning (LLM) -> output-ul e MEREU un model Pydantic aici, niciodata text liber
  - executie (tool-uri deterministe) -> primesc/produc instante ale acestor modele

Referinta: Documentatie_RO.md, sectiunile 5.1 (entitati) si 6.2 (contract reasoning).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# 1. Ticket - unitatea de baza, provenita din sursa ITSM (mock Jira-like)
# ============================================================

class Ticket(BaseModel):
    id: str = Field(..., description="Ex: INC-10234")
    source: str = Field(default="jira_mock")
    created_at: datetime
    reporter: str
    service: str
    summary: str
    description: str
    category: str
    priority: Literal["P1", "P2", "P3", "P4"]
    status: str = Field(default="Open")
    location: str

    @property
    def text_for_embedding(self) -> str:
        """summary + description concatenate, conform sectiunii 10.1 (mitigare embeddings slabe pe text scurt)."""
        return f"{self.summary}. {self.description}"


# ============================================================
# 2. IncidentCluster - rezultatul Detection Pipeline (determinist)
# ============================================================

class IncidentCluster(BaseModel):
    cluster_id: str
    ticket_ids: list[str]
    centroid_similarity: float = Field(ge=0, le=1)
    service_guess: str
    window_start: datetime
    window_end: datetime
    ticket_count: int

    @field_validator("ticket_count")
    @classmethod
    def count_matches_ids(cls, v: int, info) -> int:
        ticket_ids = info.data.get("ticket_ids")
        if ticket_ids is not None and v != len(ticket_ids):
            raise ValueError("ticket_count trebuie sa fie egal cu len(ticket_ids)")
        return v


# ============================================================
# 3. IncidentAssessment - output-ul Assessment Agent (LLM, reasoning)
# ============================================================

class IncidentAssessment(BaseModel):
    cluster_id: str
    is_major_incident_candidate: bool
    confidence: float = Field(ge=0, le=1)
    estimated_severity: Literal["SEV1", "SEV2", "SEV3", "Unknown"]
    affected_service: str
    reasoning: str = Field(..., description="Motivare textuala, cu referinte la sursele RAG")
    rag_sources: list[str] = Field(default_factory=list, description="doc_id-uri folosite din ChromaDB")
    recommended_action: Literal["propose_major_incident", "monitor", "dismiss"]


# ============================================================
# 4. CommunicationDraft - output-ul Communication Agent (LLM, reasoning)
# ============================================================

class CommunicationDraft(BaseModel):
    incident_id: str
    audience: Literal["end_users", "management"]
    subject: str
    body: str
    rag_sources: list[str] = Field(default_factory=list)
    requires_approval: bool = True


# ============================================================
# 5. MajorIncident - creat dupa aprobarea umana (human-in-the-loop #1)
# ============================================================

class MajorIncident(BaseModel):
    incident_id: str
    cluster_id: str
    status: Literal["Proposed", "Declared", "Rejected", "Resolved", "Closed"] = "Proposed"
    severity: Literal["SEV1", "SEV2", "SEV3", "Unknown"]
    declared_by: Optional[str] = None
    declared_at: Optional[datetime] = None
    root_cause_suspected: Optional[str] = None


# ============================================================
# 6. KnowledgeBaseDocument - continutul colectiilor ChromaDB (RAG)
# ============================================================

class KnowledgeBaseDocument(BaseModel):
    doc_id: str
    type: Literal["post_mortem", "runbook", "communication_template"]
    service: str
    summary: str
    content: str = Field(..., description="Corpul complet indexat in Chroma (post-mortem/runbook/template)")
    tags: list[str] = Field(default_factory=list)
    audience: Optional[Literal["end_users", "management"]] = Field(
        default=None, description="Doar pentru type=communication_template"
    )


# ============================================================
# 7. AuditLogEntry - audit trail (guvernanta), separat de trace-urile tehnice
# ============================================================

class AuditLogEntry(BaseModel):
    event_id: str
    timestamp: datetime
    actor: str = Field(..., description="ex: assessment_agent, incident_manager_07, execution_layer")
    action: str = Field(..., description="ex: propose_major_incident, approve_major_incident, send_notification")
    incident_id: Optional[str] = None
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    model: Optional[str] = None
    rag_sources: list[str] = Field(default_factory=list)
    based_on: Optional[dict] = None


# ============================================================
# 8. GroundTruthLabel - DOAR pentru date mock, folosit la KPI (acuratete clasificare, sectiunea 7)
#    Nu exista intr-un sistem real - e "cheia de corectura" pentru evaluare.
# ============================================================

class GroundTruthLabel(BaseModel):
    ticket_id: str
    incident_group_id: Optional[str] = Field(
        default=None, description="ID-ul grupului real caruia ii apartine (null daca e izolat)"
    )
    is_major_incident_ticket: bool = Field(
        ..., description="True daca ticketul face parte dintr-un burst real de Major Incident"
    )
    root_cause_id: str
    is_false_positive_trap: bool = Field(
        default=False, description="True daca ticketul e o 'capcana' (similar ca formulare, cauza diferita)"
    )