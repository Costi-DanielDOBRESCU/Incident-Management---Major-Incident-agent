"""Teste pentru logica scriptului de evaluare a retrieval-ului (fara Chroma / Ollama)."""

import pytest

from scripts.eval_retrieval import (
    analyse_case,
    is_non_incident_doc,
    is_severity_doc,
    recall_at_k,
    summarize,
)

SEV_DOC = {"doc_id": "RB-VPN-001", "content": "SEV1: 10+ users in 15 min. SEV2: 3+ users in 15 min.", "score": 0.80}
TRIAGE_DOC = {"doc_id": "RB-VPN-002", "content": "Triage steps: check gateway logs.", "score": 0.70}
NON_INC_DOC = {"doc_id": "RB-VPN-003", "content": "Known non-incident patterns: licensing per user.", "score": 0.60}
PM_DOC = {"doc_id": "PM-1", "content": "VPN outage postmortem", "score": 0.75}


def fake_query(runbooks, postmortems):
    def _query(query, collection, n, service):
        docs = runbooks if collection == "runbooks" else postmortems
        return docs[:n]
    return _query


def test_heuristics():
    assert is_severity_doc(SEV_DOC)
    assert not is_severity_doc(TRIAGE_DOC)
    assert not is_severity_doc({"content": "Only SEV3 mentioned"})
    assert is_non_incident_doc(NON_INC_DOC)
    assert not is_non_incident_doc(SEV_DOC)


def test_documents_inside_top_k():
    result = analyse_case("q", "VPN Gateway", fake_query([SEV_DOC, NON_INC_DOC, TRIAGE_DOC], [PM_DOC]), runbook_k=2)

    assert result["severity_rank"] == 1 and result["severity_in_top_k"] is True
    assert result["non_incident_rank"] == 2 and result["non_incident_in_top_k"] is True
    assert result["retrieved_runbooks"] == ["RB-VPN-001", "RB-VPN-003"]
    assert result["retrieved_postmortems"] == ["PM-1"]
    assert result["runbooks_total"] == 3


def test_non_incident_runbook_outside_top_2_is_detected():
    """Riscul din backlog: runbook-ul 'non-incident' e al 3-lea si nu ajunge la LLM cu n_results=2."""
    ranked = [SEV_DOC, TRIAGE_DOC, NON_INC_DOC]

    top2 = analyse_case("q", "VPN Gateway", fake_query(ranked, []), runbook_k=2)
    top3 = analyse_case("q", "VPN Gateway", fake_query(ranked, []), runbook_k=3)

    assert top2["non_incident_rank"] == 3
    assert top2["non_incident_in_top_k"] is False
    assert top3["non_incident_in_top_k"] is True


def test_missing_documents_give_none_ranks():
    result = analyse_case("q", "VPN Gateway", fake_query([TRIAGE_DOC], []), runbook_k=2)
    assert result["severity_rank"] is None and result["severity_in_top_k"] is False
    assert result["non_incident_rank"] is None and result["non_incident_in_top_k"] is False


def test_recall_at_k():
    ranks = [1, 2, 3, None]
    assert recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert recall_at_k(ranks, 2) == pytest.approx(0.5)
    assert recall_at_k(ranks, 3) == pytest.approx(0.75)
    assert recall_at_k(ranks, None) == pytest.approx(0.75)  # exista in serviciu
    assert recall_at_k([], 2) is None


def test_summarize():
    cases = [
        analyse_case("q", "S", fake_query([SEV_DOC, NON_INC_DOC], []), runbook_k=2),
        analyse_case("q", "S", fake_query([TRIAGE_DOC, SEV_DOC, NON_INC_DOC], []), runbook_k=2),
    ]
    s = summarize(cases)

    assert s["severity"]["@1"] == pytest.approx(0.5)
    assert s["severity"]["@2"] == pytest.approx(1.0)
    assert s["non_incident"]["@2"] == pytest.approx(0.5)
    assert s["non_incident"]["@3"] == pytest.approx(1.0)


# ---------- text real din KB (VPN) ----------

RB_VPN_002 = {"doc_id": "RB-VPN-002", "content": (
    "SEV1: Total VPN outage, all users affected, >30 min. SEV2: Partial outage, authentication degradation or severe "
    "slowness (tunnel drops, high latency) affecting multiple users (3+) within a 20-minute window. SEV3: Isolated "
    "single-user VPN issue (licensing, local config, hotel or public network blocking VPN ports). Several users with "
    "the same license or activation error after a laptop refresh is a requestwave, not an incident: see RB-VPN-009. "
    "Check the gateway dashboard before declaring."), "score": 0.7}
RB_VPN_008 = {"doc_id": "RB-VPN-008", "content": (
    "1) Open the VPN gateway dashboard. 2) Login fails for everyone: check certificate validity. Escalate to the "
    "Network Security team; for SEV1 open a bridge call. Reference: PM-2025-0117."), "score": 0.8}
RB_VPN_009 = {"doc_id": "RB-VPN-009", "content": (
    "Typical pattern: 3-8 tickets within 20 minutes with 'unable to activate VPN license'. This is a request wave, "
    "not an incident: do not declare. Route to the licensing team."), "score": 0.9}


def test_real_vpn_runbooks_are_classified_correctly():
    assert is_severity_doc(RB_VPN_002) and not is_non_incident_doc(RB_VPN_002)  # mentioneaza "not an incident", dar e severitate
    assert not is_severity_doc(RB_VPN_008) and not is_non_incident_doc(RB_VPN_008)  # triere
    assert is_non_incident_doc(RB_VPN_009) and not is_severity_doc(RB_VPN_009)


def test_severity_runbook_pushed_out_by_non_incident_runbook():
    """Situatia reala GT-002/008/016: ordinea 008, 009, 002 => runbook-ul de severitate nu ajunge in top-2."""
    result = analyse_case("q", "VPN Gateway", fake_query([RB_VPN_008, RB_VPN_009, RB_VPN_002], []), runbook_k=2)

    assert result["retrieved_runbooks"] == ["RB-VPN-008", "RB-VPN-009"]
    assert result["severity_rank"] == 3 and result["severity_in_top_k"] is False
    assert result["non_incident_rank"] == 2 and result["non_incident_in_top_k"] is True
    assert result["severity_docs"] == ["RB-VPN-002"]
    assert result["non_incident_docs"] == ["RB-VPN-009"]