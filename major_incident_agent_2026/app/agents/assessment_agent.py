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

Step 1 - Extract the EXACT numeric thresholds from the runbook above (min.
number of users/tickets, time window in minutes) for each SEV level
mentioned.

Step 2 - Directly compare the cluster's numbers (Ticket count = {cluster.ticket_count},
Window duration = {duration_minutes} minutes) against the thresholds from
Step 1. Do not dismiss a threshold just because the runbook's wording
differs slightly from the ticket wording - the NUMBERS matter, not exact
phrasing.

Step 3 - Based on the comparison in Step 2, pick estimated_severity. If the
cluster's numbers meet or exceed a SEV level's threshold, that level (or a
more severe one) is justified, even if other details are missing.

Step 4 - Decide is_major_incident_candidate and recommended_action,
consistent with Step 3 (if you picked SEV1 or SEV2, is_major_incident_candidate
must be true and recommended_action = "propose_major_incident", unless you
have a clear, explicit reason otherwise).

EXAMPLE of a correctly reasoned answer (illustrative format only, different
data, just to show the expected reasoning style):
"Step 1: RB-EXAMPLE-000 defines SEV2 at 3+ tickets within 15 min. Step 2:
this cluster has 6 tickets in 8 minutes -> exceeds the SEV2 threshold
(6>=3, 8<=15). Step 3: I choose SEV2. Step 4: is_major_incident_candidate=true,
recommended_action=propose_major_incident."

Respond STRICTLY in the required JSON format. In the "reasoning" field,
include the result of each step (1-4) explicitly, and cite the doc_ids used
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
            system="Esti un asistent tehnic precis. Raspunzi STRICT in JSON, fara text in plus.",
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