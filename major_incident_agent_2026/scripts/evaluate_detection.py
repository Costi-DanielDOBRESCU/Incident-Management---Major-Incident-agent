"""Evaluate Major Incident detection against the ground truth dataset.

This evaluator simulates the real-time behaviour of MIA: instead of
clustering all tickets globally, it processes tickets in chronological
order and, for each new ticket, only looks at the trailing 20-minute
window of tickets already seen.
"""

import json
from datetime import timedelta
from pathlib import Path

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from app.detection.clustering import cluster_similar_tickets
from app.detection.time_window import _get_created_at


BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    BASE_DIR / "app" / "data" / "mock_tickets" / "ground_truth.json"
)

TICKETS_FILE = (
    BASE_DIR / "app" / "data" / "mock_tickets" / "tickets.json"
)

MODEL_NAME = "bge-m3"
SIMILARITY_THRESHOLD = 0.75
MIN_CLUSTER_SIZE = 3
WINDOW_MINUTES = 20


def load_data() -> tuple[list[dict], list[dict]]:
    """Load ground truth and Jira-like tickets."""
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    with TICKETS_FILE.open("r", encoding="utf-8") as file:
        tickets_data = json.load(file)

    return ground_truth, tickets_data["issues"]


def build_ticket_text(ticket: dict) -> str:
    """Build the composite text representation used by the detector."""
    fields = ticket["fields"]

    summary = str(fields.get("summary", ""))
    description = str(fields.get("description", ""))

    components = ", ".join(
        str(component.get("name", ""))
        for component in fields.get("components", [])
    )

    labels = ", ".join(
        str(label)
        for label in fields.get("labels", [])
    )

    return (
        f"Summary: {summary}\n"
        f"Description: {description}\n"
        f"Component: {components}\n"
        f"Labels: {labels}"
    )


def calculate_metrics(
    ground_truth: list[dict],
    predicted_ticket_ids: set[str],
) -> tuple[int, int, int, float, float, float]:
    """Calculate TP, FP, FN, precision, recall and F1."""
    actual_major_ids = {
        item["ticket_id"]
        for item in ground_truth
        if item["is_major_incident_ticket"]
    }

    true_positives = predicted_ticket_ids & actual_major_ids
    false_positives = predicted_ticket_ids - actual_major_ids
    false_negatives = actual_major_ids - predicted_ticket_ids

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1


def run_sliding_window_clustering(
    tickets_sorted: list[dict],
    similarity_matrix: list[list[float]],
    window_minutes: int,
    threshold: float,
    min_cluster_size: int,
) -> set[frozenset[str]]:
    """
    Simulate real-time detection.

    For each ticket, in chronological order, look only at tickets already
    seen and inside the trailing time window, and run clustering on that
    subset of the precomputed similarity matrix.

    Returns every distinct cluster (as a frozenset of ticket ids) found in
    any window.
    """
    ticket_ids = [ticket["key"] for ticket in tickets_sorted]
    created_at_list = [_get_created_at(ticket) for ticket in tickets_sorted]

    found_clusters: set[frozenset[str]] = set()

    for current_index, reference_time in enumerate(created_at_list):
        window_start = reference_time - timedelta(minutes=window_minutes)

        window_indices = [
            index
            for index in range(current_index + 1)
            if window_start <= created_at_list[index] <= reference_time
        ]

        if len(window_indices) < min_cluster_size:
            continue

        sub_matrix = [
            [similarity_matrix[row][col] for col in window_indices]
            for row in window_indices
        ]

        local_clusters = cluster_similar_tickets(
            similarity_matrix=sub_matrix,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
        )

        for cluster in local_clusters:
            global_ids = frozenset(
                ticket_ids[window_indices[local_index]]
                for local_index in cluster
            )
            found_clusters.add(global_ids)

    return found_clusters


def keep_maximal_clusters(
    clusters: set[frozenset[str]],
) -> list[frozenset[str]]:
    """Drop clusters that are fully contained in a larger detected cluster."""
    clusters_by_size_desc = sorted(clusters, key=len, reverse=True)
    maximal: list[frozenset[str]] = []

    for candidate in clusters_by_size_desc:
        if not any(candidate < kept for kept in maximal):
            maximal.append(candidate)

    return maximal


def main() -> None:
    """Run the complete detection evaluation with a sliding time window."""
    settings = get_settings()
    client = Client(host=settings.ollama_base_url)

    ground_truth, tickets = load_data()

    tickets_sorted = sorted(tickets, key=_get_created_at)

    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Min cluster size: {MIN_CLUSTER_SIZE}")
    print(f"Window minutes: {WINDOW_MINUTES}")
    print()

    print(f"Ground truth tickets: {len(ground_truth)}")
    print(
        "Major incident tickets: "
        f"{sum(item['is_major_incident_ticket'] for item in ground_truth)}"
    )
    print(
        "False-positive traps: "
        f"{sum(item['is_false_positive_trap'] for item in ground_truth)}"
    )
    print()

    print("Generating embeddings...")

    embeddings = []

    for ticket in tickets_sorted:
        text = build_ticket_text(ticket)

        response = client.embed(
            model=MODEL_NAME,
            input=text,
        )

        embeddings.append(response["embeddings"][0])

    print("Calculating similarity matrix...")

    similarity_matrix = cosine_similarity(embeddings).tolist()

    print("Running sliding-window clustering (real-time simulation)...")

    found_clusters = run_sliding_window_clustering(
        tickets_sorted=tickets_sorted,
        similarity_matrix=similarity_matrix,
        window_minutes=WINDOW_MINUTES,
        threshold=SIMILARITY_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    final_clusters = keep_maximal_clusters(found_clusters)

    predicted_ticket_ids: set[str] = set()

    for cluster in final_clusters:
        predicted_ticket_ids.update(cluster)

    print()
    print(f"Distinct clusters detected across all windows: {len(final_clusters)}")
    print()

    for cluster_number, cluster in enumerate(final_clusters, start=1):
        print(f"Cluster {cluster_number}: {len(cluster)} tickets")
        print(f"  {sorted(cluster)}")

    print()
    print("Evaluation")
    print("----------")

    tp, fp, fn, precision, recall, f1 = calculate_metrics(
        ground_truth=ground_truth,
        predicted_ticket_ids=predicted_ticket_ids,
    )

    print(f"TP:        {tp}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    false_positive_traps = {
        item["ticket_id"]
        for item in ground_truth
        if item["is_false_positive_trap"]
    }

    detected_traps = predicted_ticket_ids & false_positive_traps

    print()
    print("False-positive traps")
    print("---------------------")
    print(f"Total traps:     {len(false_positive_traps)}")
    print(f"Detected traps:  {len(detected_traps)}")

    if detected_traps:
        print(f"Trap ticket IDs: {sorted(detected_traps)}")
    else:
        print("Trap ticket IDs: none")

    print()
    print("Missed major-incident tickets")
    print("-----------------------------")

    actual_major_ids = {
        item["ticket_id"]
        for item in ground_truth
        if item["is_major_incident_ticket"]
    }

    missed_major_ids = actual_major_ids - predicted_ticket_ids

    if missed_major_ids:
        print(sorted(missed_major_ids))
    else:
        print("None")


if __name__ == "__main__":
    main()