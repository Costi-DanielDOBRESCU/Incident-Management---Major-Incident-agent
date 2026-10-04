"""Tests for deterministic communication template selection."""

import json
from pathlib import Path

import pytest

from app.agents.template_selection import (
    SERVICE_SPECIFIC_USER_TEMPLATES,
    fetch_templates,
    select_template_ids,
)

KB_FILE = (
    Path(__file__).resolve().parent.parent
    / "app" / "data" / "knowledge_base" / "communication_templates.json"
)


class FakeCollection:
    """Imită ChromaDB collection.get(), cu ordine intenționat inversată față de cea cerută."""

    def __init__(self, docs: dict[str, tuple[str, dict]]):
        self._docs = docs

    def get(self, ids, include):
        present = [i for i in reversed(ids) if i in self._docs]
        return {
            "ids": present,
            "documents": [self._docs[i][0] for i in present],
            "metadatas": [self._docs[i][1] for i in present],
        }


# ---------- selectie ----------

def test_sev2_portal_gets_base_templates_only():
    assert select_template_ids("end_users", "SEV2", "Internal Portal") == ["TPL-COMM-USER-001"]
    assert select_template_ids("management", "SEV2", "Internal Portal") == ["TPL-COMM-MGMT-001"]


def test_sev1_uses_severe_templates():
    assert select_template_ids("end_users", "SEV1", "Internal Portal") == ["TPL-COMM-USER-004"]
    assert select_template_ids("management", "SEV1", "ERP System") == ["TPL-COMM-MGMT-004"]


def test_service_specific_template_is_added_for_end_users_only():
    assert select_template_ids("end_users", "SEV2", "Network/Switch") == [
        "TPL-COMM-USER-001", "TPL-COMM-USER-006",
    ]
    assert select_template_ids("management", "SEV2", "Network/Switch") == ["TPL-COMM-MGMT-001"]


def test_unknown_severity_falls_back_to_base():
    assert select_template_ids("end_users", "Unknown", "VPN Gateway") == ["TPL-COMM-USER-001"]


def test_unknown_audience_raises():
    with pytest.raises(ValueError):
        select_template_ids("press", "SEV2", "VPN Gateway")


def test_never_selects_vendor_or_closure_templates():
    forbidden = {"TPL-COMM-MGMT-005", "TPL-COMM-MGMT-006", "TPL-COMM-USER-002", "TPL-COMM-USER-003"}
    for audience in ("end_users", "management"):
        for severity in ("SEV1", "SEV2", "SEV3", "Unknown"):
            for service in list(SERVICE_SPECIFIC_USER_TEMPLATES) + ["VPN Gateway", "ERP System", "Internal Portal"]:
                assert not forbidden & set(select_template_ids(audience, severity, service))


# ---------- fetch ----------

def test_fetch_preserves_requested_order_and_skips_missing():
    collection = FakeCollection({
        "A": ("content A", {"summary": "sum A", "service": "generic"}),
        "B": ("content B", {"summary": "sum B", "service": "Network/Switch"}),
    })

    result = fetch_templates(["A", "MISSING", "B"], collection=collection)

    assert [r["doc_id"] for r in result] == ["A", "B"]
    assert result[1] == {
        "doc_id": "B", "summary": "sum B", "content": "content B",
        "service": "Network/Switch", "score": 1.0,
    }


# ---------- consistenta cu KB-ul ----------

def test_all_selectable_templates_exist_in_knowledge_base():
    kb_ids = {d["doc_id"] for d in json.loads(KB_FILE.read_text(encoding="utf-8"))}

    selectable = {"TPL-COMM-USER-001", "TPL-COMM-USER-004", "TPL-COMM-MGMT-001", "TPL-COMM-MGMT-004"}
    selectable |= set(SERVICE_SPECIFIC_USER_TEMPLATES.values())

    assert selectable <= kb_ids