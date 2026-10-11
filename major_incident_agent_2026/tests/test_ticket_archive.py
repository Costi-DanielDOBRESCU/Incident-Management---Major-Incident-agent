"""Teste pentru arhiva tichetelor unui incident."""

import pytest

from app.execution import ticket_archive


@pytest.fixture
def db(tmp_path):
    return tmp_path / "audit.db"


def _issue(key):
    return {"key": key, "fields": {"summary": f"s {key}", "description": "d", "created": "2026-10-10T10:00:00.000+0000"}}


def test_roundtrip(db):
    assert ticket_archive.archive_incident_tickets("MI-1", [_issue("A"), _issue("B")], db_path=db) == 2
    got = ticket_archive.get_incident_tickets("MI-1", db_path=db)
    assert set(got) == {"A", "B"} and got["A"]["fields"]["summary"] == "s A"


def test_archives_are_separated_per_incident(db):
    ticket_archive.archive_incident_tickets("MI-1", [_issue("A")], db_path=db)
    ticket_archive.archive_incident_tickets("MI-2", [_issue("A")], db_path=db)  # aceeasi cheie, alt incident
    assert set(ticket_archive.get_incident_tickets("MI-1", db_path=db)) == {"A"}
    assert set(ticket_archive.get_incident_tickets("MI-2", db_path=db)) == {"A"}


def test_archiving_twice_is_an_upsert(db):
    ticket_archive.archive_incident_tickets("MI-1", [_issue("A")], db_path=db)
    ticket_archive.archive_incident_tickets("MI-1", [_issue("A")], db_path=db)
    assert len(ticket_archive.get_incident_tickets("MI-1", db_path=db)) == 1


def test_unknown_incident_gives_empty_dict(db):
    assert ticket_archive.get_incident_tickets("NOPE", db_path=db) == {}