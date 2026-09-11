"""Core detection pipeline for identifying candidate incident clusters."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class DetectionCandidate:
    """Represents a group of tickets that may belong to the same incident."""

    cluster_id: str
    ticket_ids: list[str]
    detected_at: datetime


def build_detection_candidate(
    cluster_id: str,
    tickets: list[dict[str, Any]],
) -> DetectionCandidate:
    """
    Build a detection candidate from a group of tickets.

    The actual similarity and clustering algorithms will be implemented
    in the next detection components.
    """
    if not tickets:
        raise ValueError("Cannot build a detection candidate from empty tickets.")

    ticket_ids = [str(ticket["id"]) for ticket in tickets]

    return DetectionCandidate(
        cluster_id=cluster_id,
        ticket_ids=ticket_ids,
        detected_at=datetime.now(),
    )