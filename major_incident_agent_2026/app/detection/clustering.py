"""Ticket clustering logic for Major Incident detection."""

from typing import Sequence


def cluster_similar_tickets(
    similarity_matrix: Sequence[Sequence[float]],
    threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """
    Group tickets using connected components of a thresholded
    similarity graph.

    Each ticket is a node.
    An edge exists between two tickets when their similarity
    is greater than or equal to the configured threshold.

    Connected components are returned as clusters.
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

    for start_index in range(size):
        if start_index in visited:
            continue

        cluster: list[int] = []
        stack = [start_index]
        visited.add(start_index)

        while stack:
            current_index = stack.pop()
            cluster.append(current_index)

            for other_index in range(size):
                if other_index in visited:
                    continue

                if matrix[current_index][other_index] >= threshold:
                    visited.add(other_index)
                    stack.append(other_index)

        if len(cluster) >= min_cluster_size:
            clusters.append(sorted(cluster))

    return clusters