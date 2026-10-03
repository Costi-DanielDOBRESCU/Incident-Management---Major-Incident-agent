"""
app/detection/windowed_clustering.py

Detectie in stil "timp real": pentru fiecare tichet, in ordine cronologica, se grupeaza doar
tichetele din fereastra glisanta (implicit 20 min) care se termina la acel tichet. Clusterele
gasite in ferestre diferite pentru acelasi incident (care se suprapun) se unesc.

Validat cu scripts/eval_clusters.py: cu textul din app/detection/text.py, pragul 0.70 si unirea
prin suprapunere, toate cele 18 incidente din ground_truth sunt detectate intacte.

Toate functiile lucreaza cu INDICI in lista de tichete primita (la fel ca cluster_similar_tickets),
deci rezultatul se poate da direct lui build_incident_cluster().
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Sequence

from app.detection.clustering import cluster_similar_tickets
from app.detection.time_window import _get_created_at


def merge_overlapping_clusters(clusters: Sequence[Sequence[int]]) -> list[list[int]]:
    """
    Uneste clusterele care au cel putin un tichet comun (acelasi incident vazut din
    ferestre diferite). Clusterele fara tichete comune raman separate, deci doua incidente
    diferite nu se pot lipi prin aceasta operatie.
    """
    merged: list[set[int]] = []

    for cluster in clusters:
        current = set(cluster)
        remaining: list[set[int]] = []

        for existing in merged:
            if existing & current:
                current |= existing
            else:
                remaining.append(existing)

        remaining.append(current)
        merged = remaining

    return [sorted(group) for group in merged]


def detect_clusters_sliding_window(
    tickets: list[dict[str, Any]],
    similarity_matrix: Sequence[Sequence[float]],
    window_minutes: int,
    threshold: float,
    min_cluster_size: int,
) -> list[list[int]]:
    """
    Ruleaza clusteringul pe fereastra glisanta si uneste rezultatele suprapuse.

    Args:
        tickets: tichete Jira-like (cu fields.created).
        similarity_matrix: matricea de similaritate, aliniata cu `tickets`.
        window_minutes: lungimea ferestrei, in minute.
        threshold / min_cluster_size: parametrii pentru cluster_similar_tickets.

    Returns:
        Lista de clustere (liste de indici in `tickets`), ordonate dupa cel mai vechi tichet.
    """
    if window_minutes <= 0:
        raise ValueError("window_minutes must be greater than 0.")

    if len(similarity_matrix) != len(tickets):
        raise ValueError("similarity_matrix must be aligned with tickets.")

    created = {index: _get_created_at(ticket) for index, ticket in enumerate(tickets)}
    order = sorted(created, key=lambda index: created[index])

    found: set[frozenset[int]] = set()

    for position, current in enumerate(order):
        reference_time = created[current]
        window_start = reference_time - timedelta(minutes=window_minutes)

        window_indices = [
            index
            for index in order[: position + 1]
            if window_start <= created[index] <= reference_time
        ]

        if len(window_indices) < min_cluster_size:
            continue

        sub_matrix = [
            [similarity_matrix[row][col] for col in window_indices]
            for row in window_indices
        ]

        for local_cluster in cluster_similar_tickets(
            similarity_matrix=sub_matrix,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
        ):
            found.add(frozenset(window_indices[local] for local in local_cluster))

    merged = merge_overlapping_clusters([sorted(cluster) for cluster in found])

    return sorted(merged, key=lambda cluster: min(created[index] for index in cluster))