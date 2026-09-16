"""
Assessment Agent (Etapa 5) - leaga IncidentCluster (Detection, determinist)
de LLM (reasoning) si RAG (context), produce IncidentAssessment validat.

Contract respectat (doc. sectiunea 4.1/10): "LLM-ul propune, tool-urile
deterministe executa". De aceea, campurile pe care le stim deja determinist
NU sunt cerute LLM-ului, ca sa nu introducem inconsistente inutile:
  - cluster_id      -> vine direct din IncidentCluster
  - affected_service -> vine din cluster.service_guess (deja calculat de Detection)
  - rag_sources     -> vine din doc_id-urile efectiv returnate de query_knowledge_base,
                        nu e "inventat" de LLM

LLM-ul e intrebat DOAR pentru partea de judecata: is_major_incident_candidate,
confidence, estimated_severity, reasoning, recommended_action.

Referinta: Documentatie_RO.md sectiunea 4.2 (Assessment Agent), 5.1
(IncidentAssessment), 6.2 (contract reasoning), 6.4 (flux).
"""

from __future__ import annotations

from pydantic import ValidationError

from app.agents.llm_client import LlmGenerationError, generate_structured_json
from app.models.schemas import IncidentAssessment, IncidentCluster
from app.rag.chroma_client import COLLECTION_HISTORICAL, COLLECTION_RUNBOOKS
from app.rag.query import query_knowledge_base

# Campuri completate determinist de noi, NU cerute LLM-ului (vezi docstring modul).
_DETERMINISTIC_FIELDS = ("cluster_id", "affected_service", "rag_sources")


class AssessmentError(Exception):
    """Ridicata cand LLM-ul nu produce un output valid conform IncidentAssessment."""


def _llm_output_schema() -> dict:
    """
    Deriva schema JSON trimisa la Ollama din IncidentAssessment.model_json_schema(),
    eliminand campurile deterministe. Evita desincronizarea intre schema Pydantic
    si schema trimisa la LLM daca modelul se modifica ulterior.
    """
    schema = IncidentAssessment.model_json_schema()
    properties = {k: v for k, v in schema["properties"].items() if k not in _DETERMINISTIC_FIELDS}
    required = [f for f in schema.get("required", []) if f not in _DETERMINISTIC_FIELDS]

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _build_rag_query_text(cluster: IncidentCluster, ticket_summaries: list[str]) -> str:
    sample = "; ".join(ticket_summaries[:3]) if ticket_summaries else cluster.service_guess
    return f"{cluster.service_guess}: {sample}"


def _build_prompt(
    cluster: IncidentCluster,
    ticket_summaries: list[str],
    historical_context: list[dict],
    runbook_context: list[dict],
) -> str:
    """
    Construieste promptul trimis la LLM. Intentionat in ENGLEZA, desi
    codul/comentariile din proiect sunt in romana: tichetele, knowledge
    base-ul si runbook-urile sunt toate in engleza, iar un prompt mixt
    RO/EN degradeaza raationamentul unui model mic (llama3.1:8b) prin
    comutare inutila de limba intre instructiuni si date.
    """
    summaries_block = "\n".join(f"  - {s}" for s in ticket_summaries) or "  (no individual ticket details available)"

    historical_block = "\n".join(
        f"  [{doc['doc_id']}] (similarity {doc['score']}): {doc['content']}"
        for doc in historical_context
    ) or "  (no similar historical incident found)"

    runbook_block = "\n".join(
        f"  [{doc['doc_id']}] (similarity {doc['score']}): {doc['content']}"
        for doc in runbook_context
    ) or "  (no runbook found for this service)"

    duration_minutes = round((cluster.window_end - cluster.window_start).total_seconds() / 60, 1)

    return f"""You are the Assessment Agent in an IT Major Incident detection system.

You are given a cluster of correlated tickets, detected automatically
(deterministically, via semantic similarity and a time window). Assess
whether this cluster represents a genuine Major Incident candidate.

DETECTED CLUSTER (exact data, computed deterministically - use as-is,
do not recompute):
- Service: {cluster.service_guess}
- Ticket count: {cluster.ticket_count}
- Average similarity between tickets: {cluster.centroid_similarity}
- Window duration: {duration_minutes} minutes

Individual tickets (sample):
{summaries_block}

SIMILAR HISTORICAL INCIDENTS (from knowledge base):
{historical_block}

SEVERITY CRITERIA (runbook for this service):
{runbook_block}

REQUIRED METHOD (follow these steps in order, and show the result of each
step in the "reasoning" field):

Step 1 - Read the ticket content carefully and identify WHERE the root
cause is located:
  (a) LOCAL/INDIVIDUAL cause: the problem is specific to that one user's own
      device, account, license, or personal configuration (e.g. "my license
      key", "my replacement machine", "my mailbox permissions"). Multiple
      users independently hitting the SAME kind of local/individual problem
      does NOT make it a shared outage - it stays an individual-scope issue,
      no matter how many tickets pile up.
  (b) SHARED/CENTRAL cause: the problem is with a central system or service
      that many users depend on (e.g. the portal itself, the SSO provider,
      the mail server, the network switch) being down, erroring, or
      degraded for anyone who tries to use it. Several different users
      reporting the SAME central system failing is evidence FOR a shared
      outage, not against it.
  If the runbook explicitly calls out a local/individual category (licensing,
  local config, single-user, individual mailbox) AND the tickets match (a),
  that lower severity applies regardless of ticket count. If the tickets
  match (b), proceed to count-based thresholds normally.

Step 2 - Extract the EXACT numeric thresholds from the runbook above (min.
number of users/tickets, time window in minutes) for each SEV level
mentioned.

Step 3 - If Step 1 classified this as (a) LOCAL/INDIVIDUAL, that lower
severity applies and you should STOP here - do not escalate based on ticket
count. If Step 1 classified this as (b) SHARED/CENTRAL, compare the
cluster's numbers (Ticket count = {cluster.ticket_count}, Window duration =
{duration_minutes} minutes) against the thresholds from Step 2.

Step 4 - Based on Steps 1-3, pick estimated_severity.

Step 5 - Decide is_major_incident_candidate and recommended_action,
consistent with Step 4 (SEV1/SEV2 -> is_major_incident_candidate=true and
recommended_action="propose_major_incident", unless Step 1 classified this
as (a) LOCAL/INDIVIDUAL, in which case false/dismiss or monitor).

EXAMPLE A - shared/central cause (illustrative, different data):
"Step 1: tickets describe the portal/login itself failing for every
reporter - this is a SHARED/CENTRAL cause (the portal service), not a
local/individual one. Step 2: RB-EXAMPLE-000 defines SEV2 at 3+ tickets
within 15 min. Step 3: shared cause, so apply thresholds - cluster has 6
tickets in 8 minutes -> exceeds SEV2 threshold. Step 4: SEV2. Step 5:
is_major_incident_candidate=true, recommended_action=propose_major_incident."

EXAMPLE B - local/individual cause (illustrative, different data): "Step 1:
tickets describe a per-device license activation error tied to each
reporter's own new/replacement machine - this is a LOCAL/INDIVIDUAL cause
(their own license/device), matching the runbook's explicit 'isolated
single-user issue (licensing)' category, even though 3 different users hit
it. Step 2: threshold table noted for completeness. Step 3: local/individual
cause, so skip thresholds regardless of count. Step 4: SEV3. Step 5:
is_major_incident_candidate=false, recommended_action=monitor."

Respond STRICTLY in the required JSON format. In the "reasoning" field,
include the result of each step (1-5) explicitly, and cite the doc_ids used
(e.g. "per RB-VPN-002...").
"""


def assess_incident(
    cluster: IncidentCluster,
    ticket_summaries: list[str] | None = None,
) -> IncidentAssessment:
    """
    Evalueaza un IncidentCluster si produce un IncidentAssessment validat.

    Args:
        cluster: rezultatul Detection Pipeline (Etapa 3 + build_incident_cluster).
        ticket_summaries: rezumate scurte (ex. campul "summary") ale tichetelor
            din cluster, pentru context in prompt. Optional - daca lipseste,
            se foloseste doar service_guess. Apelantul (orchestrator) e cel
            care are acces la tichetele brute, nu acest modul.

    Raises:
        AssessmentError: daca LLM-ul nu raspunde sau output-ul nu valideaza
            fata de IncidentAssessment dupa completarea campurilor deterministe.
    """
    ticket_summaries = ticket_summaries or []

    query_text = _build_rag_query_text(cluster, ticket_summaries)

    historical_context = query_knowledge_base(query_text, COLLECTION_HISTORICAL, n_results=3)
    runbook_context = query_knowledge_base(
        query_text, COLLECTION_RUNBOOKS, n_results=2, service_filter=cluster.service_guess
    )

    prompt = _build_prompt(cluster, ticket_summaries, historical_context, runbook_context)
    schema = _llm_output_schema()

    try:
        llm_output = generate_structured_json(
            prompt=prompt,
            json_schema=schema,
            system="You are a precise technical assistant. Respond STRICTLY in JSON, no extra text.",
            temperature=0.1,
        )
    except LlmGenerationError as exc:
        raise AssessmentError(f"LLM-ul nu a raspuns pentru cluster {cluster.cluster_id}: {exc}") from exc

    rag_sources = [doc["doc_id"] for doc in historical_context] + [doc["doc_id"] for doc in runbook_context]

    full_output = {
        **llm_output,
        "cluster_id": cluster.cluster_id,
        "affected_service": cluster.service_guess,
        "rag_sources": rag_sources,
    }

    try:
        return IncidentAssessment(**full_output)
    except ValidationError as exc:
        raise AssessmentError(
            f"Output LLM invalid pentru cluster {cluster.cluster_id}: {exc}"
        ) from exc


if __name__ == "__main__":
    # sanity check rapid, pe clusterul real gasit anterior (GT-001, Internal Portal)
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
    test_summaries = ["Internal portal inaccessible"] * 4 + ["Portal login redirect error"]

    result = assess_incident(test_cluster, test_summaries)
    print(result.model_dump_json(indent=2))