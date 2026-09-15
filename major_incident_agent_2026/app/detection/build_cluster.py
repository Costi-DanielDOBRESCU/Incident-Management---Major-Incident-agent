"""
Conversia rezultatului brut al clustering-ului (liste de indici) in obiectul
Pydantic IncidentCluster, asa cum il asteapta Assessment Agent (Etapa 5).

Detection Pipeline (Etapa 3, inchisa) produce doar list[list[int]] - indici
in lista de tichete primita. Acest modul e "lipitura" dintre Etapa 3 si
Etapa 5: ia acei indici + tichetele brute (Jira-like) + matricea de
similaritate deja calculata, si construieste IncidentCluster complet
(centroid_similarity, service_guess, window_start/end) - conform
Documentatie_RO.md sectiunea 5.1.

Nu recalculeaza nimic ce exista deja (similarity_matrix vine din pipeline),
doar agrega.
"""

from __future__ import annotations

from app.detection.time_window import _get_created_at
from app.models.schemas import IncidentCluster


def _centroid_similarity(similarity_matrix: list[list[float]], indices: list[int]) -> float:
    """
    Media similaritatilor pairwise intre toate tichetele clusterului
    (triunghiul superior al submatricei), NU similaritatea fata de un
    singur centroid geometric - suficient pentru MVP si coerent cu
    pragul de clustering (care e deja pairwise, in clustering.py).
    """
    if len(indices) < 2:
        return 1.0

    pairs = [
        similarity_matrix[i][j]
        for a, i in enumerate(indices)
        for j in indices[a + 1:]
    ]
    return sum(pairs) / len(pairs)


def _service_guess(tickets: list[dict], indices: list[int]) -> str:
    """
    Serviciul majoritar in cluster, extras din ticket['fields']['components'].
    Daca sunt voturi egale, primul serviciu intalnit (ordinea tichetelor) castiga.
    """
    counts: dict[str, int] = {}
    order: list[str] = []

    for i in indices:
        components = tickets[i]["fields"].get("components") or []
        service = components[0]["name"] if components else "Unknown"
        if service not in counts:
            order.append(service)
        counts[service] = counts.get(service, 0) + 1

    return max(order, key=lambda s: counts[s])


def build_incident_cluster(
    tickets: list[dict],
    similarity_matrix: list[list[float]],
    indices: list[int],
    sequence: int,
) -> IncidentCluster:
    """
    Construieste un IncidentCluster validat Pydantic dintr-un cluster brut
    (indici in `tickets`), asa cum e produs de cluster_similar_tickets().

    Args:
        tickets: lista completa de tichete Jira-like (aceeasi lista folosita
            la calculul similarity_matrix - indicii trebuie sa corespunda).
        similarity_matrix: matricea de similaritate cosinus, deja calculata
            (calculate_similarity_matrix).
        indices: indicii tichetelor din acest cluster specific.
        sequence: numar secvential al clusterului in cadrul batch-ului curent
            de detectie, folosit pentru un cluster_id lizibil si unic
            (ex. "CL-20260819-003"). Apelantul (orchestrator/pipeline) e
            responsabil sa dea valori unice per rulare.

    Returns:
        IncidentCluster validat, gata pentru Assessment Agent.
    """
    if len(indices) < 2:
        raise ValueError("Un cluster trebuie sa aiba cel putin 2 tichete.")

    cluster_tickets = [tickets[i] for i in indices]
    created_ats = [_get_created_at(t) for t in cluster_tickets]

    window_start = min(created_ats)
    window_end = max(created_ats)

    cluster_id = f"CL-{window_start:%Y%m%d}-{sequence:03d}"

    return IncidentCluster(
        cluster_id=cluster_id,
        ticket_ids=[t["key"] for t in cluster_tickets],
        centroid_similarity=round(_centroid_similarity(similarity_matrix, indices), 4),
        service_guess=_service_guess(tickets, indices),
        window_start=window_start,
        window_end=window_end,
        ticket_count=len(indices),
    )