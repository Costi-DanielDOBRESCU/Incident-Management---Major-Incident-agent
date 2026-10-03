"""Detection pipeline connecting ingestion, embeddings, similarity and clustering."""

from datetime import datetime
from typing import Any, Callable

from app.config import get_settings
from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.detection.text import build_ticket_text
from app.detection.time_window import filter_tickets_by_time_window
from app.detection.windowed_clustering import detect_clusters_sliding_window
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


def _has_timestamps(tickets: list[dict[str, Any]]) -> bool:
    return all("created" in (ticket.get("fields") or {}) for ticket in tickets)


def detect_clusters_with_matrix(
    tickets: list[dict[str, Any]],
    embed: Callable[[str], list[float]] | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Generate ticket embeddings, calculate cosine similarity and identify candidate clusters.

    The text embedded for each ticket is the composite text from app.detection.text
    (summary, description, component, labels). When tickets carry creation timestamps,
    clustering runs on a sliding time window (settings.clustering_window_minutes) and
    overlapping clusters are merged; otherwise it falls back to a single global clustering.

    Args:
        tickets: Jira-like tickets.
        embed: optional embedding function (e.g. a cached version); defaults to create_embedding.

    Returns:
        (clusters, similarity_matrix): clusters are lists of indices into `tickets`;
        the matrix is aligned with `tickets` so it can be passed to build_incident_cluster().
    """
    if not tickets:
        return [], []

    settings = get_settings()
    embed_fn = embed or create_embedding

    texts = [build_ticket_text(ticket) for ticket in tickets]
    embeddings = [embed_fn(text) for text in texts]
    similarity_matrix = calculate_similarity_matrix(embeddings)

    if _has_timestamps(tickets):
        clusters = detect_clusters_sliding_window(
            tickets=tickets,
            similarity_matrix=similarity_matrix,
            window_minutes=settings.clustering_window_minutes,
            threshold=settings.similarity_threshold,
            min_cluster_size=settings.min_tickets_per_cluster,
        )
    else:
        clusters = cluster_similar_tickets(
            similarity_matrix=similarity_matrix,
            threshold=settings.similarity_threshold,
            min_cluster_size=settings.min_tickets_per_cluster,
        )

    return clusters, similarity_matrix


def detect_clusters(
    tickets: list[dict[str, Any]],
) -> list[list[int]]:
    """
    Generate ticket embeddings, calculate cosine similarity,
    and identify candidate clusters.
    """
    clusters, _ = detect_clusters_with_matrix(tickets)
    return clusters