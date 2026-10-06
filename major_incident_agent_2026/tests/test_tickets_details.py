"""Teste pentru citirea unui tichet complet din SQLite (folosit de UI pentru detalii)."""

from app.ingestion import ticket_store


def _issue(key="INC-1"):
    return {
        "key": key,
        "fields": {
            "summary": "VPN down",
            "description": "Users in building A cannot connect to the VPN since 09:55.",
            "priority": {"name": "P2 - High"},
            "status": {"name": "Open"},
            "components": [{"name": "VPN Gateway"}],
            "reporter": {"name": "alice"},
            "labels": ["vpn", "remote"],
            "location": "Building A",
            "created": "2026-10-06T10:00:00.000+0000",
        },
    }


def test_get_raw_issue_returns_full_ticket(tmp_path):
    db = tmp_path / "tickets.db"
    ticket_store.init_db(db)
    ticket_store.upsert_tickets([_issue("INC-1"), _issue("INC-2")], db_path=db)

    issue = ticket_store.get_raw_issue("INC-1", db_path=db)

    assert issue["key"] == "INC-1"
    assert issue["fields"]["description"].startswith("Users in building A")
    assert issue["fields"]["labels"] == ["vpn", "remote"]
    assert issue["fields"]["location"] == "Building A"


def test_get_raw_issue_unknown_key_is_none(tmp_path):
    db = tmp_path / "tickets.db"
    ticket_store.init_db(db)
    assert ticket_store.get_raw_issue("NOPE", db_path=db) is None