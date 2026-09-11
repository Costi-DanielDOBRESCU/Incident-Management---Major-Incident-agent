"""Cosine similarity calculations for ticket embeddings."""

from typing import Sequence

from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_matrix(
    embeddings: Sequence[Sequence[float]],
) -> list[list[float]]:
    """
    Calculate the cosine similarity matrix for a collection of embeddings.

    Each row and column corresponds to one ticket.
    """
    if not embeddings:
        return []

    matrix = cosine_similarity(embeddings)

    return matrix.tolist()