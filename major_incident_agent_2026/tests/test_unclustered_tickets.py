"""Teste pentru tichetele fara cluster: cel mai apropiat cluster si starea marcata de operator."""

import numpy as np
import pytest

from app.detection.unclustered import find_unclustered
from app.execution import audit_store, ticket_reviews


def _ticket(key, service="VPN Gateway", created="2026-10-06T10:00:00.000+0000"):
    return {
        "key": key,
        "fields": {
            "summary": f"summary {key}",
            "created": created,
            "components": [{"name": service}],
        },
    }


# ---------- find_unclustered ----------

def test_only_unclustered_tickets_are_returned_sorted_by_similarity():
    tickets = [_ticket(f"T-{i}") for i in range(5)]
    # T-0,T-1,T-2 = cluster; T-3 si T-4 in afara lui
    sim = np.array([
        [1.0, 0.9, 0.9, 0.2, 0.60],
        [0.9, 1.0, 0.9, 0.3, 0.55],
        [0.9, 0.9, 1.0, 0.4, 0.50],
        [0.2, 0.3, 0.4, 1.0, 0.1],
        [0.6, 0.55, 0.5, 0.1, 1.0],
    ])

    result = find_unclustered(tickets, [[0, 1, 2]], sim, ["CL-1"])

    assert [r["key"] for r in result] == ["T-4", "T-3"]  # cel mai apropiat primul
    assert result[0]["nearest_cluster_id"] == "CL-1"
    assert result[0]["nearest_similarity"] == pytest.approx(0.60)
    assert result[1]["nearest_similarity"] == pytest.approx(0.40)


def test_nearest_cluster_is_chosen_among_several():
    tickets = [_ticket(f"T-{i}") for i in range(7)]
    sim = np.zeros((7, 7))
    sim[6, 0] = 0.30   # fata de clusterul A
    sim[6, 4] = 0.65   # fata de clusterul B
    result = find_unclustered(tickets, [[0, 1, 2], [3, 4, 5]], sim, ["CL-A", "CL-B"])

    assert len(result) == 1
    assert result[0]["key"] == "T-6"
    assert result[0]["nearest_cluster_id"] == "CL-B"
    assert result[0]["nearest_similarity"] == pytest.approx(0.65)


def test_no_clusters_means_no_nearest_info():
    tickets = [_ticket("T-0"), _ticket("T-1", service="Email")]
    result = find_unclustered(tickets, [], np.eye(2), [])

    assert [r["key"] for r in result] == ["T-0", "T-1"]
    assert all(r["nearest_cluster_id"] is None and r["nearest_similarity"] is None for r in result)
    assert result[1]["service"] == "Email"


def test_accepts_plain_lists_as_matrix():
    tickets = [_ticket("T-0"), _ticket("T-1")]
    result = find_unclustered(tickets, [[0]], [[1.0, 0.4], [0.4, 1.0]], ["CL-1"])
    assert result[0]["nearest_similarity"] == pytest.approx(0.4)


# ---------- ticket_reviews ----------

@pytest.fixture
def db(tmp_path):
    return tmp_path / "audit.db"


def test_set_and_list_review(db):
    ticket_reviews.set_review("T-1", "2026-10-06T10:00:00", "handled", "alice", db_path=db)

    reviews = ticket_reviews.list_reviews(db_path=db)
    assert reviews[("T-1", "2026-10-06T10:00:00")]["status"] == "handled"
    assert reviews[("T-1", "2026-10-06T10:00:00")]["reviewed_by"] == "alice"


def test_status_can_change_and_unreviewed_removes_it(db):
    key = ("T-1", "2026-10-06T10:00:00")
    ticket_reviews.set_review(*key, "watch", "alice", db_path=db)
    ticket_reviews.set_review(*key, "handled", "bob", db_path=db)
    assert ticket_reviews.list_reviews(db_path=db)[key]["status"] == "handled"
    assert ticket_reviews.list_reviews(db_path=db)[key]["reviewed_by"] == "bob"

    ticket_reviews.set_review(*key, "unreviewed", "bob", db_path=db)
    assert ticket_reviews.list_reviews(db_path=db) == {}


def test_same_key_with_different_created_is_a_different_ticket(db):
    ticket_reviews.set_review("T-1", "2026-10-06T10:00:00", "handled", "alice", db_path=db)
    reviews = ticket_reviews.list_reviews(db_path=db)
    assert ("T-1", "2026-10-06T11:30:00") not in reviews


def test_review_is_audited(db):
    ticket_reviews.set_review("T-1", "2026-10-06T10:00:00", "watch", "alice", db_path=db)

    (event,) = audit_store.list_audit_events(db_path=db)
    assert event["actor"] == "alice"
    assert event["action"] == "review_unclustered_ticket"
    assert event["input_ref"] == "T-1"
    assert event["output_ref"] == "watch"


def test_unknown_status_is_rejected(db):
    with pytest.raises(ValueError):
        ticket_reviews.set_review("T-1", "2026-10-06T10:00:00", "bogus", "alice", db_path=db)