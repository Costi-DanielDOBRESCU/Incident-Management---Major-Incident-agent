"""Teste pentru retentia tichetelor fara cluster."""

import time

import pytest

from app.ingestion import retention, ticket_store


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "tickets.db"
    ticket_store.init_db(path)
    return path


def _issue(key, minutes_ago, now):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000+0000", time.gmtime(now - minutes_ago * 60))
    return {
        "key": key,
        "fields": {
            "summary": f"s {key}",
            "description": "d",
            "created": ts,
            "priority": {"name": "P2"},
            "status": {"name": "Open"},
            "components": [{"name": "VPN Gateway"}],
        },
    }


def _keys(db):
    return {r["key"] for r in ticket_store.list_tickets(db_path=db)}


def test_unreviewed_and_watch_are_kept_handled_and_clustered_are_dropped(db):
    now = time.time()
    issues = [_issue(k, 5, now) for k in ("C1", "U1", "W1", "H1")]
    ticket_store.upsert_tickets(issues, db_path=db)
    created = {i["key"]: i["fields"]["created"] for i in issues}
    reviews = {
        ("W1", created["W1"]): {"status": "watch"},
        ("H1", created["H1"]): {"status": "handled"},
    }
    res = retention.prune_tickets({"C1"}, 120, reviews, now=now, db_path=db)
    assert _keys(db) == {"U1", "W1"}
    assert res["dropped_clustered"] == 1 and res["dropped_handled"] == 1


def test_expired_unclustered_tickets_are_dropped(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("OLD", 300, now), _issue("NEW", 10, now)], db_path=db)
    res = retention.prune_tickets(set(), 120, {}, now=now, db_path=db)
    assert _keys(db) == {"NEW"}
    assert res["dropped_expired"] == 1


def test_watermark_is_not_touched(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("A", 5, now)], db_path=db)
    before = ticket_store.get_watermark(db)
    retention.prune_tickets({"A"}, 120, {}, now=now, db_path=db)
    assert ticket_store.get_watermark(db) == before


def test_new_ticket_with_same_key_replaces_carried_over_one(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("INC-1", 60, now)], db_path=db)
    new = ticket_store.upsert_tickets([_issue("INC-1", 1, now)], db_path=db)
    assert len(new) == 1  # tichetul nou nu e blocat de cel pastrat
    assert ticket_store.count_tickets(db) == 1


def test_same_key_same_created_is_still_deduplicated(db):
    now = time.time()
    issue = _issue("INC-1", 5, now)
    ticket_store.upsert_tickets([issue], db_path=db)
    assert ticket_store.upsert_tickets([issue], db_path=db) == []