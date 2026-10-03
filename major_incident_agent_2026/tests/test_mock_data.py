"""
tests/test_mock_data.py

Teste pytest pentru datele mock (generate_mock_tickets.py v2 + knowledge base extins)
si pentru mock_jira_api.py.

Rulare:
    pytest tests/test_mock_data.py -v
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.ingestion.mock_jira_api import fetch_tickets

# ---------------------------------------------------------------------------
# Căi + praguri
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

MIN_TICKETS_PER_CLUSTER = 3

SERVICES = {
    "VPN Gateway", "Email/Exchange", "ERP System", "Network/Switch",
    "Cloud Storage", "Internal Portal", "Print Services",
}
KB_SERVICES = SERVICES | {"generic"}
TEMPLATE_PLACEHOLDERS = {
    "service", "time", "next_update", "ticket_count", "window",
    "suspected_cause", "severity", "incident_id", "root_cause", "duration",
}


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


def _service_of(issue: dict) -> str:
    return issue["fields"]["components"][0]["name"]


# ---------------------------------------------------------------------------
# 1. Volum de date
# ---------------------------------------------------------------------------

def test_ticket_volume_in_expected_range(tickets_data):
    issues = tickets_data["issues"]
    assert 500 <= len(issues) <= 1000
    # "total" trebuie să reflecte corect numărul de tichete din fișier
    assert tickets_data["total"] == len(issues)


# ---------------------------------------------------------------------------
# 2. Forma Jira-like
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


def test_ticket_content_is_realistic(tickets_data):
    issues = tickets_data["issues"]

    # prioritățile folosesc nume corecte, fără combinații contradictorii
    allowed_priorities = {"P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"}
    assert {i["fields"]["priority"]["name"] for i in issues} <= allowed_priorities

    # toate cele 7 servicii sunt prezente
    assert {_service_of(i) for i in issues} == SERVICES

    # textele nu sunt șabloane repetate: variație mare în rezumate și descrieri
    summaries = {i["fields"]["summary"] for i in issues}
    descriptions = {i["fields"]["description"] for i in issues}
    assert len(summaries) >= 0.35 * len(issues)
    assert len(descriptions) >= 0.55 * len(issues)

    # fără placeholdere nerezolvate
    for i in issues:
        assert "{" not in i["fields"]["summary"]
        assert "{" not in i["fields"]["description"]

    # tichetele sunt în ordine cronologică
    created = [i["fields"]["created"] for i in issues]
    assert created == sorted(created)


# ---------------------------------------------------------------------------
# 3. Ground truth — grupuri (incidente reale)
# ---------------------------------------------------------------------------

def test_ground_truth_groups_meet_cluster_threshold(tickets_data, ground_truth_data):
    # ground_truth.json trebuie să acopere exact aceleași tichete ca tickets.json
    ticket_keys = {issue["key"] for issue in tickets_data["issues"]}
    gt_keys = {entry["ticket_id"] for entry in ground_truth_data}
    assert gt_keys == ticket_keys

    service_by_key = {i["key"]: _service_of(i) for i in tickets_data["issues"]}

    groups: dict[str, list[dict]] = {}
    for entry in ground_truth_data:
        gid = entry["incident_group_id"]
        if gid is not None:
            groups.setdefault(gid, []).append(entry)

    # 18 incidente reale (GT-001 ... GT-018)
    assert len(groups) == 18

    for gid, members in groups.items():
        # fiecare grup respectă pragul minim de tichete/cluster
        assert len(members) >= MIN_TICKETS_PER_CLUSTER
        # toate tichetele dintr-un grup sunt marcate ca major incident
        assert all(m["is_major_incident_ticket"] for m in members)
        # aceeași cauză reală în cadrul grupului
        assert len({m["root_cause_id"] for m in members}) == 1
        # și același serviciu
        assert len({service_by_key[m["ticket_id"]] for m in members}) == 1
        # niciun membru de grup nu poate fi și trap
        assert not any(m["is_false_positive_trap"] for m in members)

    # incidentele acoperă toate cele 7 servicii
    covered = {service_by_key[members[0]["ticket_id"]] for members in groups.values()}
    assert covered == SERVICES


# ---------------------------------------------------------------------------
# 4. Ground truth — capcane de fals-pozitiv
# ---------------------------------------------------------------------------

def test_ground_truth_false_positive_traps_are_isolated(ground_truth_data):
    traps = [e for e in ground_truth_data if e["is_false_positive_trap"]]

    # trebuie să existe capcane, dar rămân o minoritate clară
    assert 10 <= len(traps) <= 80

    for trap in traps:
        # un trap NU face parte dintr-un cluster ground truth și nu e marcat major incident
        assert trap["incident_group_id"] is None
        assert trap["is_major_incident_ticket"] is False

    # tichetele izolate (fără grup, fără trap) + grupate + traps acoperă tot setul
    grouped = sum(1 for e in ground_truth_data if e["incident_group_id"] is not None)
    isolated = sum(
        1 for e in ground_truth_data
        if e["incident_group_id"] is None and not e["is_false_positive_trap"]
    )
    assert grouped + isolated + len(traps) == len(ground_truth_data)

    # zgomotul de fundal domină setul, ca într-un helpdesk real
    assert isolated > grouped


def test_noise_does_not_form_accidental_clusters(tickets_data, ground_truth_data):
    """Orice 3 tichete consecutive ale aceluiași serviciu în 20 de minute trebuie să fie
    dintr-un incident sau dintr-o capcană intenționată, niciodată zgomot de fundal."""
    from datetime import datetime

    gt = {e["ticket_id"]: e for e in ground_truth_data}
    per_service: dict[str, list[tuple[datetime, str]]] = {}
    for i in tickets_data["issues"]:
        ts = datetime.strptime(i["fields"]["created"], "%Y-%m-%dT%H:%M:%S.000+0000")
        per_service.setdefault(_service_of(i), []).append((ts, i["key"]))

    for service, items in per_service.items():
        items.sort()
        for a, b, c in zip(items, items[1:], items[2:]):
            if (c[0] - a[0]).total_seconds() <= 20 * 60:
                for _, key in (a, b, c):
                    entry = gt[key]
                    assert entry["incident_group_id"] or entry["is_false_positive_trap"], (
                        f"Cluster accidental de zgomot în {service} în jurul {a[0]}"
                    )


# ---------------------------------------------------------------------------
# 5. Knowledge base — cele 3 colecții
# ---------------------------------------------------------------------------

def test_knowledge_base_collections(kb_documents):
    # 20 post-mortemuri + 24 runbook-uri + 14 șabloane de comunicare
    counts = Counter(doc["type"] for doc in kb_documents)
    assert counts == {"post_mortem": 20, "runbook": 24, "communication_template": 14}
    assert len(kb_documents) == 58

    required_keys = {"doc_id", "type", "service", "summary", "content", "tags"}
    doc_ids = set()
    for doc in kb_documents:
        assert required_keys.issubset(doc.keys())
        assert doc["doc_id"] not in doc_ids, f"doc_id duplicat: {doc['doc_id']}"
        doc_ids.add(doc["doc_id"])
        assert doc["type"] in {"post_mortem", "runbook", "communication_template"}
        assert doc["service"] in KB_SERVICES
        assert doc["content"].strip() and doc["tags"]


def test_knowledge_base_covers_every_service(kb_documents):
    for service in SERVICES:
        types = {d["type"] for d in kb_documents if d["service"] == service}
        assert {"post_mortem", "runbook"} <= types, f"Acoperire incompletă pentru {service}"


def test_knowledge_base_templates_and_references(kb_documents):
    ids = {d["doc_id"] for d in kb_documents}

    for doc in kb_documents:
        if doc["type"] == "communication_template":
            assert doc["audience"] in {"end_users", "management"}
            used = set(re.findall(r"\{(\w+)\}", doc["content"]))
            assert used <= TEMPLATE_PLACEHOLDERS, f"{doc['doc_id']}: {used - TEMPLATE_PLACEHOLDERS}"

        # referințele către alte documente trebuie să existe
        for ref in re.findall(r"\b(?:PM-\d{4}-\d{4}|RB-[A-Z]+-\d{3}|TPL-COMM-[A-Z]+-\d{3})\b", doc["content"]):
            assert ref in ids, f"{doc['doc_id']} face referire la {ref}, care nu există"


# ---------------------------------------------------------------------------
# 6. Mock API — search + filtrare `since` (tool fetch_tickets)
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