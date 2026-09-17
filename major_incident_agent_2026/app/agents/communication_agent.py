"""
Communication Agent (Etapa 6) - genereaza CommunicationDraft pe baza unui
MajorIncident aprobat, folosind RAG (colectia communication_templates) si
LLM (Ollama), apelat o data per audienta (end_users, management) - conform
Documentatie_RO.md, sectiunea 4.2 (Communication Agent), 5.1
(CommunicationDraft), 6.4 (Pas 7-8).

Contract respectat (acelasi ca la Assessment Agent, sectiunea 4.1/10):
campurile deterministe NU sunt cerute LLM-ului:
  - incident_id     -> vine din MajorIncident.incident_id
  - audience        -> data ca parametru de apelant, nu decisa de LLM
  - rag_sources     -> vine din doc_id-urile efectiv returnate de query_knowledge_base
  - requires_approval -> mereu True (doc. sectiunea 10: aprobare umana
                          obligatorie pentru comunicarea externa)

LLM-ul genereaza DOAR subject si body.

Promptul e in ENGLEZA (vezi motivatia in assessment_agent.py si
STATUS_PROIECT_MIA6.md, gotcha 12) - date + template-uri sunt in engleza.

temperature=0.7 la apelul LLM (diferit de Assessment Agent, care foloseste
0.1): aici e generare de text natural, nu clasificare - varietate/naturalete
> determinism strict.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import ValidationError

from app.agents.llm_client import LlmGenerationError, generate_structured_json
from app.models.schemas import CommunicationDraft, IncidentAssessment, IncidentCluster, MajorIncident
from app.rag.chroma_client import COLLECTION_TEMPLATES
from app.rag.query import query_knowledge_base

_DETERMINISTIC_FIELDS = ("incident_id", "audience", "rag_sources", "requires_approval")

# Cat timp adaugam la window_end pentru un "next update" implicit, cand LLM-ul
# trebuie sa mentioneze un ETA - decizie determinista, nu lasata pe seama LLM-ului
# (evita ore inventate/inconsistente intre cele 2 audiente).
_NEXT_UPDATE_DELAY_MINUTES = 60


class CommunicationError(Exception):
    """Ridicata cand LLM-ul nu produce un output valid conform CommunicationDraft."""


def _llm_output_schema() -> dict:
    """Deriva schema JSON trimisa la Ollama din CommunicationDraft, ca la assessment_agent.py."""
    schema = CommunicationDraft.model_json_schema()
    properties = {k: v for k, v in schema["properties"].items() if k not in _DETERMINISTIC_FIELDS}
    required = [f for f in schema.get("required", []) if f not in _DETERMINISTIC_FIELDS]
    return {"type": "object", "properties": properties, "required": required}


def _build_rag_query_text(cluster: IncidentCluster, audience: str) -> str:
    # Simetric cu exemplul din documentatie: "VPN outage communication template".
    return f"{cluster.service_guess} outage communication template {audience}"


# def _build_prompt(
#     incident: MajorIncident,
#     assessment: IncidentAssessment,
#     cluster: IncidentCluster,
#     audience: Literal["end_users", "management"],
#     template_context: list[dict],
# ) -> str:
#     template_block = "\n".join(
#         f"  [{doc['doc_id']}] (similarity {doc['score']}): {doc['content']}"
#         for doc in template_context
#     ) or "  (no template found for this audience)"

#     next_update = cluster.window_end + timedelta(minutes=_NEXT_UPDATE_DELAY_MINUTES)
#     duration_minutes = round((cluster.window_end - cluster.window_start).total_seconds() / 60, 1)

#     audience_instructions = {
#         "end_users": (
#             "Write a short, plain-language message for END USERS. Do not include "
#             "internal details (root cause hypotheses, ticket counts, doc_ids, "
#             "severity codes). Focus on: what is affected, that it's being worked "
#             "on, and when the next update will come."
#         ),
#         "management": (
#             "Write a concise, factual message for MANAGEMENT. Include the "
#             "concrete numbers (ticket count, time window, severity) and the "
#             "suspected cause. This audience expects data, not reassurance."
#         ),
#     }[audience]

#     return f"""You are the Communication Agent in an IT Major Incident management system.

# A Major Incident has been APPROVED by a human (human-in-the-loop already
# happened - do not question whether this should be declared, only draft the
# communication).

# INCIDENT (exact data, computed deterministically - use as-is):
# - Incident ID: {incident.incident_id}
# - Service: {cluster.service_guess}
# - Severity: {incident.severity}
# - Ticket count: {cluster.ticket_count}
# - Incident started at: {cluster.window_start:%H:%M} UTC
# - Time window duration: {duration_minutes} minutes
# - Suspected root cause: {incident.root_cause_suspected or assessment.reasoning}
# - Next update ETA: {next_update:%H:%M} UTC

# TARGET AUDIENCE: {audience}
# {audience_instructions}

# RELEVANT TEMPLATE(S) FOR THIS AUDIENCE (from knowledge base - match this
# tone and structure, adapt the placeholder content to the incident above,
# do NOT leave literal placeholders like "{{service}}" in your output):
# {template_block}

# If the template uses a phrase like "since ~{{time}}", use the "Incident
# started at" clock time above for that - NOT the duration in minutes.

# Respond STRICTLY in the required JSON format, with fields "subject" and
# "body" only. The body should read as a finished message ready to send, not
# a template.
# """

import re

def _clean_root_cause(assessment: IncidentAssessment) -> str:
    """ Curata complet prefixele de reasoning ale LLM-ului. """
    reasoning = assessment.reasoning or ""
    
    # Daca contine "SHARED/CENTRAL" sau "Step 1", oferim un rezumat curat direct
    if "SHARED/CENTRAL" in reasoning or "Step 1" in reasoning:
        return f"Service outage/degradation affecting {assessment.affected_service}"
    
    # Eliminam eventualele ramasite de 'Step X:'
    cleaned = re.sub(r"^Step\s*\d+:?\s*", "", reasoning, flags=re.IGNORECASE).strip()
    first_sentence = cleaned.split(".")[0]
    
    return first_sentence if len(first_sentence) < 100 else f"Service issue on {assessment.affected_service}"


def _build_prompt(
    incident: MajorIncident,
    assessment: IncidentAssessment,
    cluster: IncidentCluster,
    audience: Literal["end_users", "management"],
    template_context: list[dict],
) -> str:
    template_block = "\n".join(
        f"  [{doc['doc_id']}] (similarity {doc['score']}): {doc['content']}"
        for doc in template_context
    ) or "  (no template found for this audience)"

    # FIX 1: Calculam "Next update" relativ la momentul executiei curente (datetime.now)
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    next_update = now_utc + timedelta(minutes=_NEXT_UPDATE_DELAY_MINUTES)
    
    duration_minutes = round((cluster.window_end - cluster.window_start).total_seconds() / 60, 1)

    # FIX 2: Ignoram textul de debug (Step 1-5, SHARED/CENTRAL) pentru a nu fi preluat de LLM
    raw_cause = incident.root_cause_suspected or assessment.reasoning or ""
    if "Step 1" in raw_cause or "SHARED/CENTRAL" in raw_cause or not raw_cause:
        suspected_cause = f"Central service disruption affecting {cluster.service_guess}"
    else:
        suspected_cause = _clean_root_cause(assessment)

    audience_instructions = {
        "end_users": (
            "Write a short, plain-language message for END USERS. Do not include "
            "internal details (root cause hypotheses, ticket counts, doc_ids, "
            "severity codes). Focus on: what is affected, that it's being worked "
            "on, and when the next update will come."
        ),
        "management": (
            "Write a concise, factual message for MANAGEMENT. Include the "
            "concrete numbers (ticket count, time window, severity) and a short "
            "1-sentence summary of the suspected cause. Do NOT output LLM step-by-step reasoning, "
            "step numbers, or placeholders like {TPL-COMM-MGMT-001} in the final text."
            "CRITICAL: Replace ALL template placeholders like {ticket_count} or {window} with their actual numeric values from INCIDENT DATA."
        ),
    }[audience]

    return f"""You are the Communication Agent in an IT Major Incident management system.

A Major Incident has been APPROVED by a human.

INCIDENT DATA:
- Incident ID: {incident.incident_id}
- Service: {cluster.service_guess}
- Severity: {incident.severity}
- Ticket count: {cluster.ticket_count}
- Incident started at: {cluster.window_start:%H:%M} UTC
- Time window duration: {duration_minutes} minutes
- Suspected root cause: {suspected_cause}
- Next update ETA: {next_update:%H:%M} UTC

TARGET AUDIENCE: {audience}
{audience_instructions}

RELEVANT TEMPLATES (from knowledge base - match tone, adapt content, DO NOT leave placeholders like {{service}}):
{template_block}

Respond STRICTLY in JSON format with fields "subject" and "body". The body must be a final, clean, ready-to-send text.
"""


def generate_communication(
    incident: MajorIncident,
    assessment: IncidentAssessment,
    cluster: IncidentCluster,
    audience: Literal["end_users", "management"],
) -> CommunicationDraft:
    """
    Genereaza un CommunicationDraft pentru o singura audienta. Se apeleaza
    de 2 ori (o data per audienta) de catre orchestrator/UI, conform
    Documentatie_RO.md Pas 8.

    Args:
        incident: MajorIncident deja aprobat (human-in-the-loop trecut).
        assessment: IncidentAssessment original (reasoning, ca fallback
            pentru root_cause_suspected daca acesta e None pe incident).
        cluster: IncidentCluster original (service, ticket_count, fereastra).
        audience: "end_users" sau "management".

    Raises:
        CommunicationError: daca LLM-ul nu raspunde sau output-ul nu
            valideaza fata de CommunicationDraft.
    """
    query_text = _build_rag_query_text(cluster, audience)
    template_context = query_knowledge_base(
        query_text, COLLECTION_TEMPLATES, n_results=2, audience_filter=audience
    )

    prompt = _build_prompt(incident, assessment, cluster, audience, template_context)
    schema = _llm_output_schema()

    try:
        llm_output = generate_structured_json(
            prompt=prompt,
            json_schema=schema,
            system="You are a precise, professional technical communications assistant. Respond STRICTLY in JSON, no extra text.",
            temperature=0.7,
        )
    except LlmGenerationError as exc:
        raise CommunicationError(
            f"LLM-ul nu a raspuns pentru incident {incident.incident_id} ({audience}): {exc}"
        ) from exc

    full_output = {
        **llm_output,
        "incident_id": incident.incident_id,
        "audience": audience,
        "rag_sources": [doc["doc_id"] for doc in template_context],
        "requires_approval": True,
    }

    try:
        return CommunicationDraft(**full_output)
    except ValidationError as exc:
        raise CommunicationError(
            f"Output LLM invalid pentru incident {incident.incident_id} ({audience}): {exc}"
        ) from exc


if __name__ == "__main__":
    # sanity check rapid, pe cazul GT-001 (Internal Portal), aprobat manual
    from datetime import datetime, timezone

    test_cluster = IncidentCluster(
        cluster_id="CL-TEST-001",
        ticket_ids=["INC-10021", "INC-10022", "INC-10023", "INC-10024", "INC-10025"],
        centroid_similarity=1.0,
        service_guess="Internal Portal",
        window_start=datetime(2026, 8, 1, 23, 34, 17, tzinfo=timezone.utc),
        window_end=datetime(2026, 8, 1, 23, 43, 41, tzinfo=timezone.utc),
        ticket_count=5,
    )
    test_assessment = IncidentAssessment(
        cluster_id="CL-TEST-001",
        is_major_incident_candidate=True,
        confidence=0.8,
        estimated_severity="SEV2",
        affected_service="Internal Portal",
        reasoning="5 tickets in 9.4 minutes exceed the SEV2 threshold (3+ users within 15 min) per RB-PORTAL-006.",
        rag_sources=["RB-PORTAL-006", "PM-2025-0176"],
        recommended_action="propose_major_incident",
    )
    test_incident = MajorIncident(
        incident_id="MI-CL-TEST-001",
        cluster_id="CL-TEST-001",
        status="Declared",
        severity="SEV2",
        declared_by="demo_user",
        declared_at=datetime.now(timezone.utc),
        root_cause_suspected="Portal login redirect loop, suspected SSO misconfiguration",
    )

    for aud in ("end_users", "management"):
        draft = generate_communication(test_incident, test_assessment, test_cluster, aud)
        print(f"\n--- {aud} ---")
        print(draft.model_dump_json(indent=2))