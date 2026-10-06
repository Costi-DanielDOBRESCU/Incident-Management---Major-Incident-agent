"""Tests for the incident memory (human decisions stored for RAG), using a fake Chroma collection."""

import pytest

from app.execution import incident_memory


class FakeCollection:
    def __init__(self, distance=0.2):
        self.docs = {}
        self.distance = distance
        self.last_query = None

    def count(self):
        return len(self.docs)

    def upsert(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.docs[i] = (d, m)

    def query(self, query_texts, n_results, where, include):
        self.last_query = {"query": query_texts[0], "n": n_results, "where": where}
        items = [
            (i, d, m) for i, (d, m) in self.docs.items()
            if not where or m.get("service") == where.get("service")
        ][:n_results]
        return {
            "ids": [[i for i, _, _ in items]],
            "documents": [[d for _, d, _ in items]],
            "metadatas": [[m for _, _, m in items]],
            "distances": [[self.distance for _ in items]],
        }


def _decision(**overrides):
    base = {
        "incident_id": "MI-CL-1",
        "service": "VPN Gateway",
        "severity": "SEV2",
        "outcome": "declared",
        "reason": "",
        "decided_by": "alice",
        "decided_at": "2026-08-03T10:00:00+00:00",
        "summaries": ["Cannot connect to VPN", "VPN authentication failing", "Cannot connect to VPN"],
    }
    base.update(overrides)
    return base


def test_memory_text_starts_like_the_assessment_query():
    text = incident_memory.build_memory_text(_decision())

    assert text.startswith("VPN Gateway: Cannot connect to VPN; VPN authentication failing.")
    assert text.count("Cannot connect to VPN") == 1  # duplicatele se elimina
    assert "a major incident (SEV2)" in text
    assert "no post-mortem yet" in text


def test_rejection_text_carries_the_reason():
    text = incident_memory.build_memory_text(_decision(outcome="rejected", reason="Licensing wave after laptop refresh"))

    assert "REJECTED" in text
    assert "Reason: Licensing wave after laptop refresh" in text


def test_remember_decision_stores_validated_metadata():
    collection = FakeCollection()

    doc_id = incident_memory.remember_decision(_decision(), collection=collection)

    assert doc_id == "MEM-MI-CL-1"
    _, metadata = collection.docs[doc_id]
    assert metadata["outcome"] == "declared"
    assert metadata["human_validated"] is True
    assert metadata["post_mortem"] == "pending"
    assert metadata["service"] == "VPN Gateway"


def test_remember_decision_is_idempotent():
    collection = FakeCollection()

    incident_memory.remember_decision(_decision(), collection=collection)
    incident_memory.remember_decision(_decision(outcome="rejected", reason="changed mind"), collection=collection)

    assert len(collection.docs) == 1
    assert collection.docs["MEM-MI-CL-1"][1]["outcome"] == "rejected"


def test_only_human_decisions_are_remembered():
    with pytest.raises(ValueError):
        incident_memory.remember_decision(_decision(outcome="proposed"), collection=FakeCollection())


def test_search_filters_by_service_and_returns_scored_results():
    collection = FakeCollection(distance=0.2)
    incident_memory.remember_decision(_decision(), collection=collection)
    incident_memory.remember_decision(_decision(incident_id="MI-CL-2", service="ERP System"), collection=collection)

    results = incident_memory.search_memory("VPN Gateway: cannot connect", "VPN Gateway", collection=collection)

    assert [r["doc_id"] for r in results] == ["MEM-MI-CL-1"]
    assert results[0]["score"] == 0.8
    assert results[0]["outcome"] == "declared"
    assert collection.last_query["where"] == {"service": "VPN Gateway"}


def test_search_drops_weak_matches():
    collection = FakeCollection(distance=0.6)  # similaritate 0.4 < MIN_SCORE
    incident_memory.remember_decision(_decision(), collection=collection)

    assert incident_memory.search_memory("anything", "VPN Gateway", collection=collection) == []


def test_search_on_empty_memory_or_blank_query_returns_nothing():
    assert incident_memory.search_memory("x", "VPN Gateway", collection=FakeCollection()) == []
    assert incident_memory.search_memory("   ", "VPN Gateway", collection=FakeCollection()) == []


def test_search_never_raises_when_the_store_is_unavailable():
    class Broken(FakeCollection):
        def count(self):
            raise RuntimeError("chroma down")

    assert incident_memory.search_memory("x", "VPN Gateway", collection=Broken()) == []