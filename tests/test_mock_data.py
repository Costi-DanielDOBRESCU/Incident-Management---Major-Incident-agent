"""
tests/test_mock_data.py

Etapa 1-2, fișier 4/4 — 6 teste pytest pentru datele mock generate de
scripts/generate_mock_data.py și pentru mock_jira_api.py.

Rulare:
    pytest tests/test_mock_data.py -v
"""

import json
from pathlib import Path

import pytest

from app.ingestion.mock_jira_api import fetch_tickets

# ---------------------------------------------------------------------------
# Căi + praguri (secțiunea 6.5 din documentație)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MOCK_TICKETS_DIR = BASE_DIR / "app" / "data" / "mock_tickets"
KB_DIR = BASE_DIR / "app" / "data" / "knowledge_base"

TICKETS_FILE = MOCK_TICKETS_DIR / "tickets.json"
GROUND_TRUTH_FILE = MOCK_TICKETS_DIR / "ground_truth.json"

KB_FILES = [
    KB_DIR / "communication_templates.json",
    KB_DIR / "historical_major_incidents.json",
    KB_DIR / "runbooks.json",
]

MIN_TICKETS_PER_CLUSTER = 3  # secțiunea 6.5 a documentației


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tickets_data() -> dict:
    with open(TICKETS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ground_truth_data() -> list[dict]:
    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def kb_documents() -> list[dict]:
    docs = []
    for path in KB_FILES:
        with open(path, encoding="utf-8") as f:
            docs.extend(json.load(f))
    return docs


# ---------------------------------------------------------------------------
# 1. Volum de date (secțiunea 5.2: 200-500 tichete)
# ---------------------------------------------------------------------------

def test_ticket_volume_in_expected_range(tickets_data):
    issues = tickets_data["issues"]
    assert 200 <= len(issues) <= 500
    # "total" trebuie să reflecte corect numărul de tichete din fișier
    assert tickets_data["total"] == len(issues)


# ---------------------------------------------------------------------------
# 2. Forma Jira-like (secțiunea 5.4)
# ---------------------------------------------------------------------------

def test_tickets_match_jira_like_shape(tickets_data):
    issues = tickets_data["issues"]
    assert len(issues) > 0

    required_field_keys = {
        "summary", "description", "created", "priority",
        "status", "components", "reporter", "labels", "location",
    }

    for issue in issues:
        assert "key" in issue
        assert issue["key"].startswith("INC-")
        assert "fields" in issue

        fields = issue["fields"]
        assert required_field_keys.issubset(fields.keys())

        assert isinstance(fields["priority"], dict) and "name" in fields["priority"]
        assert isinstance(fields["status"], dict) and "name" in fields["status"]
        assert isinstance(fields["components"], list) and len(fields["components"]) >= 1
        assert "name" in fields["components"][0]
        assert isinstance(fields["reporter"], dict) and "name" in fields["reporter"]
        assert isinstance(fields["labels"], list)

    # cheile trebuie să fie unice
    keys = [issue["key"] for issue in issues]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 3. Ground truth — grupuri (burst-uri reale, secțiunea 5.2 + 6.5)
# ---------------------------------------------------------------------------

def test_ground_truth_groups_meet_cluster_threshold(tickets_data, ground_truth_data):
    # ground_truth.json trebuie să acopere exact aceleași tichete ca tickets.json
    ticket_keys = {issue["key"] for issue in tickets_data["issues"]}
    gt_keys = {entry["ticket_id"] for entry in ground_truth_data}
    assert gt_keys == ticket_keys

    groups: dict[str, list[dict]] = {}
    for entry in ground_truth_data:
        gid = entry["incident_group_id"]
        if gid is not None:
            groups.setdefault(gid, []).append(entry)

    # cele 10 grupuri ground truth din secțiunea 5.2
    assert len(groups) == 10

    for gid, members in groups.items():
        # fiecare grup trebuie să respecte pragul minim de tichete/cluster
        assert len(members) >= MIN_TICKETS_PER_CLUSTER
        # toate tichetele dintr-un grup trebuie marcate ca major incident
        assert all(m["is_major_incident_ticket"] for m in members)
        # și trebuie să aibă aceeași cauză reală (root_cause_id) în cadrul grupului
        root_causes = {m["root_cause_id"] for m in members}
        assert len(root_causes) == 1
        # niciun membru de grup nu poate fi și trap
        assert not any(m["is_false_positive_trap"] for m in members)


# ---------------------------------------------------------------------------
# 4. Ground truth — false-positive traps (secțiunea 5.2)
# ---------------------------------------------------------------------------

def test_ground_truth_false_positive_traps_are_isolated(ground_truth_data):
    traps = [e for e in ground_truth_data if e["is_false_positive_trap"]]

    # trebuie să existe cel puțin câteva traps, dar nu prea multe (sunt cazuri intenționate rare)
    assert 1 <= len(traps) <= 30

    for trap in traps:
        # un trap NU face parte dintr-un cluster ground truth și nu e marcat major incident
        assert trap["incident_group_id"] is None
        assert trap["is_major_incident_ticket"] is False

    # tichetele izolate (fără grup, fără trap) + grupate + traps trebuie să acopere tot setul
    grouped = sum(1 for e in ground_truth_data if e["incident_group_id"] is not None)
    isolated = sum(
        1 for e in ground_truth_data
        if e["incident_group_id"] is None and not e["is_false_positive_trap"]
    )
    assert grouped + isolated + len(traps) == len(ground_truth_data)


# ---------------------------------------------------------------------------
# 5. Knowledge base — cele 3 colecții (secțiunea 5.3)
# ---------------------------------------------------------------------------

def test_knowledge_base_collections(kb_documents):
    assert len(kb_documents) == 18  # 4 + 7 + 7, secțiunea 5.3

    required_keys = {"doc_id", "type", "service", "summary", "content", "tags"}
    doc_ids = set()
    for doc in kb_documents:
        assert required_keys.issubset(doc.keys())
        assert doc["doc_id"] not in doc_ids, f"doc_id duplicat: {doc['doc_id']}"
        doc_ids.add(doc["doc_id"])
        assert doc["type"] in {"post_mortem", "runbook", "communication_template"}


# ---------------------------------------------------------------------------
# 6. Mock API — search + filtrare `since` (secțiunea 5.4, tool fetch_tickets)
# ---------------------------------------------------------------------------

def test_mock_api_search_and_since_filter(tickets_data):
    full = fetch_tickets(limit=500)
    assert full["total"] == len(tickets_data["issues"])

    # respectă limita cerută
    limited = fetch_tickets(limit=5)
    assert len(limited["issues"]) == 5
    assert limited["total"] == full["total"]  # total = total match, nu pagina

    # filtrare `since`: alegem un timestamp la mijlocul intervalului
    all_created = sorted(
        issue["fields"]["created"] for issue in tickets_data["issues"]
    )
    midpoint = all_created[len(all_created) // 2]

    filtered = fetch_tickets(since=midpoint, limit=500)
    assert 0 < filtered["total"] < full["total"]

    for issue in filtered["issues"]:
        assert issue["fields"]["created"] >= midpoint