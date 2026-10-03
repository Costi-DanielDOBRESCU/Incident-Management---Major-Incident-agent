"""
app/detection/text.py

Sursa UNICA pentru textul trimis la embeddings in detectie (varianta C, validata cu
scripts/sweep_detection.py si scripts/eval_clusters.py):

    Summary + Description + Component + Labels

Folosit de pipeline-ul de detectie, de UI si de scripturile de evaluare, ca sa nu existe
versiuni diferite ale textului.
"""

from __future__ import annotations

from typing import Any


def build_ticket_text(ticket: dict[str, Any]) -> str:
    """Build the composite text representation used by the detector."""
    fields = ticket["fields"]

    summary = str(fields.get("summary", ""))
    description = str(fields.get("description", ""))

    components = ", ".join(
        str(component.get("name", ""))
        for component in (fields.get("components") or [])
    )

    labels = ", ".join(
        str(label)
        for label in (fields.get("labels") or [])
    )

    return (
        f"Summary: {summary}\n"
        f"Description: {description}\n"
        f"Component: {components}\n"
        f"Labels: {labels}"
    )