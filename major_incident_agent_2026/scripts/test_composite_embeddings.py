import json
from pathlib import Path

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from app.detection.clustering import cluster_similar_tickets


BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = BASE_DIR / "app" / "data" / "mock_tickets" / "ground_truth.json"
TICKETS_FILE = BASE_DIR / "app" / "data" / "mock_tickets" / "tickets.json"

MODEL_NAME = "bge-m3"
SIMILARITY_THRESHOLD = 0.75
MIN_CLUSTER_SIZE = 3


def build_ticket_text(ticket: dict) -> str:
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


def load_data():
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    with TICKETS_FILE.open("r", encoding="utf-8") as file:
        tickets_data = json.load(file)

    tickets = tickets_data["issues"]

    return ground_truth, tickets


def main():
    settings = get_settings()
    client = Client(host=settings.ollama_base_url)

    ground_truth, tickets = load_data()

    tickets_by_id = {
        ticket["key"]: ticket
        for ticket in tickets
    }

    groups = {}

    for item in ground_truth:
        group_id = item["incident_group_id"]

        if not item["is_major_incident_ticket"]:
            continue

        groups.setdefault(group_id, []).append(item["ticket_id"])

    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Min cluster size: {MIN_CLUSTER_SIZE}")
    print()

    for group_id, ticket_ids in sorted(groups.items()):
        group_tickets = [
            tickets_by_id[ticket_id]
            for ticket_id in ticket_ids
            if ticket_id in tickets_by_id
        ]

        texts = [
            build_ticket_text(ticket)
            for ticket in group_tickets
        ]

        embeddings = []

        for text in texts:
            response = client.embed(
                model=MODEL_NAME,
                input=text,
            )

            embeddings.append(response["embeddings"][0])

        similarity_matrix = cosine_similarity(embeddings).tolist()

        clusters = cluster_similar_tickets(
            similarity_matrix=similarity_matrix,
            threshold=SIMILARITY_THRESHOLD,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )

        detected_count = sum(len(cluster) for cluster in clusters)

        print(
            f"{group_id}: "
            f"expected={len(group_tickets)}, "
            f"detected={detected_count}, "
            f"clusters={clusters}"
        )


if __name__ == "__main__":
    main()