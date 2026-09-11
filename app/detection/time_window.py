"""Time-window filtering for the Major Incident detection pipeline."""

from datetime import datetime, timedelta
from typing import Any


def _get_created_at(ticket: dict[str, Any]) -> datetime:
    """Extract and parse the Jira-like ticket creation timestamp."""
    created_at = ticket["fields"]["created"]

    if isinstance(created_at, datetime):
        return created_at

    value = str(created_at).strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    if value.endswith("+0000"):
        value = value[:-5] + "+00:00"

    return datetime.fromisoformat(value)


def filter_tickets_by_time_window(
    tickets: list[dict[str, Any]],
    reference_time: datetime,
    window_minutes: int = 20,
) -> list[dict[str, Any]]:
    """
    Return Jira-like tickets created within the configured time window.

    The window starts at reference_time - window_minutes and ends
    at reference_time, inclusive.
    """
    if window_minutes <= 0:
        raise ValueError("window_minutes must be greater than 0.")

    window_start = reference_time - timedelta(minutes=window_minutes)

    filtered_tickets = []

    for ticket in tickets:
        created_at = _get_created_at(ticket)

        if window_start <= created_at <= reference_time:
            filtered_tickets.append(ticket)

    return filtered_tickets