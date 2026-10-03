"""Tests for sliding-window clustering, cluster merging and the detection text."""

from datetime import datetime, timedelta, timezone

import pytest

from app.detection.text import build_ticket_text
from app.detection.windowed_clustering import (
    detect_clusters_sliding_window,
    merge_overlapping_clusters,
)

BASE = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)


def _ticket(key: str, minutes: float) -> dict:
    created = (BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    return {"key": key, "fields": {"summary": key, "created": created}}


def _matrix(size: int, similar_groups: list[list[int]]) -> list[list[float]]:
    """Identity matrix where tickets in the same group have similarity 0.9."""
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for group in similar_groups:
        for i in group:
            for j in group:
                if i != j:
                    matrix[i][j] = 0.9
    return matrix


# ---------- merge_overlapping_clusters ----------

def test_merge_joins_clusters_that_share_a_ticket():
    assert merge_overlapping_clusters([[0, 1, 2], [2, 3, 4]]) == [[0, 1, 2, 3, 4]]


def test_merge_keeps_disjoint_clusters_separate():
    result = merge_overlapping_clusters([[0, 1, 2], [3, 4, 5]])
    assert sorted(result) == [[0, 1, 2], [3, 4, 5]]


def test_merge_absorbs_subsets():
    assert merge_overlapping_clusters([[0, 1, 2], [0, 1, 2, 3]]) == [[0, 1, 2, 3]]


def test_merge_chains_through_multiple_clusters():
    assert merge_overlapping_clusters([[0, 1], [2, 3], [1, 2]]) == [[0, 1, 2, 3]]


def test_merge_empty():
    assert merge_overlapping_clusters([]) == []


# ---------- detect_clusters_sliding_window ----------

def test_tickets_far_apart_in_time_do_not_cluster():
    tickets = [_ticket("A", 0), _ticket("B", 40), _ticket("C", 80)]
    matrix = _matrix(3, [[0, 1, 2]])

    result = detect_clusters_sliding_window(tickets, matrix, 20, 0.7, 3)

    assert result == []


def test_tickets_inside_window_cluster():
    tickets = [_ticket("A", 0), _ticket("B", 5), _ticket("C", 10)]
    matrix = _matrix(3, [[0, 1, 2]])

    assert detect_clusters_sliding_window(tickets, matrix, 20, 0.7, 3) == [[0, 1, 2]]


def test_long_incident_is_unified_across_windows():
    # 6 tickets over 30 minutes: no single 20-minute window contains all of them
    tickets = [_ticket(str(i), minute) for i, minute in enumerate([0, 6, 12, 18, 24, 30])]
    matrix = _matrix(6, [[0, 1, 2, 3, 4, 5]])

    assert detect_clusters_sliding_window(tickets, matrix, 20, 0.7, 3) == [[0, 1, 2, 3, 4, 5]]


def test_two_simultaneous_incidents_stay_separate():
    tickets = [_ticket(str(i), minute) for i, minute in enumerate([0, 1, 2, 3, 4, 5])]
    matrix = _matrix(6, [[0, 2, 4], [1, 3, 5]])

    result = detect_clusters_sliding_window(tickets, matrix, 20, 0.7, 3)

    assert sorted(result) == [[0, 2, 4], [1, 3, 5]]


def test_unsorted_input_returns_indices_into_original_list():
    tickets = [_ticket("late", 10), _ticket("early", 0), _ticket("mid", 5)]
    matrix = _matrix(3, [[0, 1, 2]])

    assert detect_clusters_sliding_window(tickets, matrix, 20, 0.7, 3) == [[0, 1, 2]]


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        detect_clusters_sliding_window([_ticket("A", 0)], [[1.0]], 0, 0.7, 3)


def test_misaligned_matrix_raises():
    with pytest.raises(ValueError):
        detect_clusters_sliding_window([_ticket("A", 0), _ticket("B", 1)], [[1.0]], 20, 0.7, 3)


# ---------- build_ticket_text ----------

def test_ticket_text_contains_all_fields():
    ticket = {
        "key": "INC-1",
        "fields": {
            "summary": "VPN down",
            "description": "Cannot connect since 09:00",
            "components": [{"name": "VPN Gateway"}],
            "labels": ["network"],
        },
    }

    text = build_ticket_text(ticket)

    assert text == (
        "Summary: VPN down\n"
        "Description: Cannot connect since 09:00\n"
        "Component: VPN Gateway\n"
        "Labels: network"
    )


def test_ticket_text_tolerates_missing_fields():
    text = build_ticket_text({"fields": {"summary": "Only summary"}})

    assert text.startswith("Summary: Only summary\nDescription: \n")
    assert text.endswith("Component: \nLabels: ")