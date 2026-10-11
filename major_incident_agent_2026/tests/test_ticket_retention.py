"""Teste pentru retentia tichetelor si legarea lor de incidente (tickets.db)."""

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


# ---------- punctul 5: clusterele respinse nu pierd tichetele ----------

def test_tickets_not_linked_to_a_declared_incident_are_kept(db):
    """Tichetele din clustere respinse / abandonate sunt tichete individuale: raman (nerevizuite)."""
    now = time.time()
    ticket_store.upsert_tickets([_issue("REJ-1", 5, now), _issue("REJ-2", 5, now)], db_path=db)
    res = retention.prune_tickets(120, {}, now=now, db_path=db)
    assert _keys(db) == {"REJ-1", "REJ-2"}
    assert res["dropped_linked"] == 0


def test_linked_and_handled_are_dropped_unreviewed_and_watch_are_kept(db):
    now = time.time()
    issues = [_issue(k, 5, now) for k in ("L1", "U1", "W1", "H1")]
    ticket_store.upsert_tickets(issues, db_path=db)
    ticket_store.link_tickets(["L1"], "MI-1", "Linked-MajorIncident", cluster_id="CL-1", db_path=db)
    created = {i["key"]: i["fields"]["created"] for i in issues}
    reviews = {
        ("W1", created["W1"]): {"status": "watch"},
        ("H1", created["H1"]): {"status": "handled"},
    }
    res = retention.prune_tickets(120, reviews, now=now, db_path=db)
    assert _keys(db) == {"U1", "W1"}
    assert res["dropped_linked"] == 1 and res["dropped_handled"] == 1


def test_expired_tickets_are_dropped(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("OLD", 300, now), _issue("NEW", 10, now)], db_path=db)
    res = retention.prune_tickets(120, {}, now=now, db_path=db)
    assert _keys(db) == {"NEW"}
    assert res["dropped_expired"] == 1


def test_watermark_is_not_touched(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("A", 5, now)], db_path=db)
    before = ticket_store.get_watermark(db)
    retention.prune_tickets(120, {}, now=now, db_path=db)
    assert ticket_store.get_watermark(db) == before


# ---------- punctul 6: tichetele legate nu reintra la detectie ----------

def test_linked_tickets_are_excluded_from_detection_snapshot(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("A", 5, now), _issue("B", 4, now), _issue("C", 3, now)], db_path=db)
    ticket_store.link_tickets(["A", "B"], "MI-1", "Linked-MajorIncident", db_path=db)
    assert [i["key"] for i in ticket_store.list_raw_issues(db_path=db)] == ["A", "B", "C"]
    assert [i["key"] for i in ticket_store.list_raw_issues(db_path=db, exclude_linked=True)] == ["C"]


# ---------- punctul 1: legatura si statusul sunt vizibile in tabel ----------

def test_link_tickets_sets_parent_status_and_cluster(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("A", 5, now), _issue("B", 4, now)], db_path=db)
    n = ticket_store.link_tickets(["A"], "MI-1", "Linked-MajorIncident", cluster_id="CL-9", db_path=db)
    assert n == 1
    rows = {r["key"]: r for r in ticket_store.list_tickets(db_path=db)}
    assert rows["A"]["parent_incident_id"] == "MI-1"
    assert rows["A"]["ticket_status"] == "Linked-MajorIncident"
    assert rows["A"]["cluster_id"] == "CL-9"
    assert rows["B"]["parent_incident_id"] is None


def test_old_database_without_new_columns_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE tickets (key TEXT PRIMARY KEY, created TEXT NOT NULL, created_ts REAL NOT NULL, "
        "service TEXT, priority TEXT, summary TEXT, description TEXT, raw_json TEXT NOT NULL, "
        "fetched_at TEXT NOT NULL, cluster_id TEXT);"
    )
    conn.commit()
    conn.close()
    ticket_store.init_db(path)
    assert ticket_store.list_tickets(db_path=path) == []  # SELECT pe coloanele noi nu mai da eroare
    assert ticket_store.link_tickets(["X"], "MI-1", "S", db_path=path) == 0


# ---------- chei care se repeta ----------

def test_new_ticket_with_same_key_replaces_carried_over_one_and_resets_link(db):
    now = time.time()
    ticket_store.upsert_tickets([_issue("INC-1", 60, now)], db_path=db)
    ticket_store.link_tickets(["INC-1"], "MI-OLD", "Linked-MajorIncident", db_path=db)
    new = ticket_store.upsert_tickets([_issue("INC-1", 1, now)], db_path=db)
    assert len(new) == 1  # tichetul nou nu e blocat de cel vechi
    row = ticket_store.list_tickets(db_path=db)[0]
    assert row["parent_incident_id"] is None  # legatura veche nu se transfera pe tichetul nou


def test_same_key_same_created_is_still_deduplicated(db):
    now = time.time()
    issue = _issue("INC-1", 5, now)
    ticket_store.upsert_tickets([issue], db_path=db)
    assert ticket_store.upsert_tickets([issue], db_path=db) == []