"""Detection pipeline connecting ingestion, embeddings, similarity and clustering."""

from datetime import datetime
from typing import Any

from app.config import get_settings
from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.detection.time_window import filter_tickets_by_time_window
from app.ingestion.mock_jira_api import fetch_tickets


def get_recent_tickets(
    reference_time: datetime,
    window_minutes: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """
    Retrieve tickets from the mock Jira API and keep only recent tickets.
    """
    settings = get_settings()

    if window_minutes is None:
        window_minutes = settings.clustering_window_minutes

    result = fetch_tickets(limit=limit)
    tickets = result["issues"]

    return filter_tickets_by_time_window(
        tickets=tickets,
        reference_time=reference_time,
        window_minutes=window_minutes,
    )


def detect_clusters(
    tickets: list[dict[str, Any]],
) -> list[list[int]]:
    """
    Generate ticket embeddings, calculate cosine similarity,
    and identify candidate clusters.
    """
    if not tickets:
        return []

    settings = get_settings()

    texts = [
        str(ticket["fields"].get("summary", ""))
        for ticket in tickets
    ]

    embeddings = [create_embedding(text) for text in texts]

    similarity_matrix = calculate_similarity_matrix(embeddings)

    return cluster_similar_tickets(
        similarity_matrix=similarity_matrix,
        threshold=settings.similarity_threshold,
        min_cluster_size=settings.min_tickets_per_cluster,
    )