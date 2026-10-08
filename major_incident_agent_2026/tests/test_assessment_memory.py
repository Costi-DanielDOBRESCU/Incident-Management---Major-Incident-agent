"""Teste pentru folosirea memoriei de decizii in Assessment Agent (fara LLM / Chroma reale)."""

from datetime import datetime, timezone

import pytest

from app.agents import assessment_agent as aa
from app.models.schemas import IncidentCluster


def _cluster():
    now = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
    return IncidentCluster(
        cluster_id="CL-1",
        ticket_ids=["T-1", "T-2", "T-3"],
        centroid_similarity=0.9,
        service_guess="VPN Gateway",
        window_start=now,
        window_end=now,
        ticket_count=3,
    )


MEMORY_HIT = {
    "doc_id": "MEM-MI-CL-9-abc123",
    "content": "VPN Gateway: Cannot connect. A human REJECTED this cluster as NOT a major incident. Reason: licensing.",
    "outcome": "rejected",
    "severity": "SEV2",
    "score": 0.81,
}


@pytest.fixture
def calls(monkeypatch):
    """Stub-uri pentru RAG, memorie si LLM; `calls` retine ce s-a apelat."""
    calls = {"prompt": None, "memory_args": None, "memory_called": False}

    monkeypatch.setattr(
        aa, "query_knowledge_base",
        lambda q, collection, n_results, service_filter=None: [{"doc_id": f"DOC-{collection}", "content": "x", "score": 0.9}],
    )

    def fake_memory(query, service=None, **kwargs):
        calls["memory_called"] = True
        calls["memory_args"] = (query, service)
        return calls.get("memory_result", [])

    monkeypatch.setattr(aa, "search_memory", fake_memory)

    def fake_llm(prompt, json_schema, system=None, **kwargs):
        calls["prompt"] = prompt
        return {
            "is_major_incident_candidate": True,
            "confidence": 0.9,
            "estimated_severity": "SEV2",
            "reasoning": "shared cause",
            "recommended_action": "propose_major_incident",
        }

    monkeypatch.setattr(aa, "generate_structured_json", fake_llm)
    return calls


def test_memory_is_added_to_prompt_and_sources(calls):
    calls["memory_result"] = [MEMORY_HIT]

    result = aa.assess_incident(_cluster(), ["Cannot connect to VPN"])

    assert "PAST HUMAN DECISIONS ON SIMILAR CLUSTERS" in calls["prompt"]
    assert "informative only; current tickets and runbook rules decide" in calls["prompt"]
    assert "[MEM-MI-CL-9-abc123] (human decision: rejected SEV2; similarity 0.81)" in calls["prompt"]
    assert "Never let them override" in calls["prompt"]
    assert result.rag_sources[-1] == "MEM-MI-CL-9-abc123"
    assert calls["memory_args"][1] == "VPN Gateway"  # cautare doar pe acelasi serviciu


def test_memory_section_comes_before_runbook_rules(calls):
    calls["memory_result"] = [MEMORY_HIT]
    aa.assess_incident(_cluster(), ["Cannot connect to VPN"])

    prompt = calls["prompt"]
    assert prompt.index("HISTORICAL CONTEXT") < prompt.index("PAST HUMAN DECISIONS") < prompt.index("RUNBOOK SEVERITY RULES")


def test_empty_memory_leaves_prompt_unchanged(calls):
    calls["memory_result"] = []
    result = aa.assess_incident(_cluster(), ["Cannot connect to VPN"])

    assert "PAST HUMAN DECISIONS" not in calls["prompt"]
    assert not any(s.startswith("MEM-") for s in result.rag_sources)


def test_use_memory_false_skips_the_lookup(calls):
    calls["memory_result"] = [MEMORY_HIT]
    result = aa.assess_incident(_cluster(), ["Cannot connect to VPN"], use_memory=False)

    assert calls["memory_called"] is False
    assert "PAST HUMAN DECISIONS" not in calls["prompt"]
    assert not any(s.startswith("MEM-") for s in result.rag_sources)


def test_prompt_without_memory_is_identical_with_and_without_the_feature(calls):
    """Reproductibilitate: fara memorie, promptul e cel de dinainte (aceeasi structura, fara sectiune goala)."""
    calls["memory_result"] = []
    aa.assess_incident(_cluster(), ["Cannot connect to VPN"], use_memory=True)
    with_feature = calls["prompt"]
    aa.assess_incident(_cluster(), ["Cannot connect to VPN"], use_memory=False)

    assert with_feature == calls["prompt"]


def test_all_service_runbooks_are_retrieved(monkeypatch, calls):
    """Regresie: cu n_results=2, runbook-ul de severitate lipsea in 5/18 incidente. Un serviciu are 3 runbook-uri."""
    seen = {}

    def spy(q, collection, n_results, service_filter=None):
        seen[collection] = n_results
        return []

    monkeypatch.setattr(aa, "query_knowledge_base", spy)
    aa.assess_incident(_cluster(), ["Cannot connect to VPN"])

    assert seen["runbooks"] >= 3