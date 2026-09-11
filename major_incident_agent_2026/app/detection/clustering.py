"""Ticket clustering logic for Major Incident detection."""

from typing import Sequence


def cluster_similar_tickets(
    similarity_matrix: Sequence[Sequence[float]],
    threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """
    Group tickets whose pairwise similarity meets the configured threshold.

    Each row and column in the similarity matrix represents one ticket.
    The returned values are ticket indexes.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1.")

    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2.")

    matrix = [list(row) for row in similarity_matrix]

    if not matrix:
        return []

    size = len(matrix)

    if any(len(row) != size for row in matrix):
        raise ValueError("similarity_matrix must be square.")

    visited: set[int] = set()
    clusters: list[list[int]] = []

    for ticket_index in range(size):
        if ticket_index in visited:
            continue

        cluster = [ticket_index]

        for other_index in range(ticket_index + 1, size):
            if other_index in visited:
                continue

            if matrix[ticket_index][other_index] >= threshold:
                cluster.append(other_index)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
            visited.update(cluster)

    return clusters