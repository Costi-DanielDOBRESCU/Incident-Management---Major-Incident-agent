"""Evaluate Major Incident detection against the ground truth dataset."""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = (
    BASE_DIR / "app" / "data" / "mock_tickets" / "ground_truth.json"
)


def load_ground_truth() -> list[dict]:
    """Load the ground truth dataset."""
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Load and inspect the ground truth dataset."""
    ground_truth = load_ground_truth()

    print(f"Ground truth tickets: {len(ground_truth)}")

    major_tickets = [
        item
        for item in ground_truth
        if item["is_major_incident_ticket"]
    ]

    false_positive_traps = [
        item
        for item in ground_truth
        if item["is_false_positive_trap"]
    ]

    incident_groups = {
        item["incident_group_id"]
        for item in ground_truth
        if item["incident_group_id"] is not None
    }

    print(f"Major incident tickets: {len(major_tickets)}")
    print(f"False-positive traps: {len(false_positive_traps)}")
    print(f"Incident groups: {len(incident_groups)}")


if __name__ == "__main__":
    main()