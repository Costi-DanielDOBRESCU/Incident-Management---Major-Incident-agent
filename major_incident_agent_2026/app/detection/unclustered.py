"""
Tichete care nu intra in niciun cluster, cu informatia 'cat de aproape e de cel mai apropiat cluster'.

Doar informativ: nu modifica clusterele si nu influenteaza detectia. Ajuta operatorul sa vada daca un
tichet apropiat de prag ar putea fi legat de un incident.
"""

from typing import Any


def _service(ticket: dict[str, Any]) -> str:
    components = ticket["fields"].get("components") or []
    return components[0]["name"] if components else "Unknown"


def find_unclustered(
    tickets: list[dict[str, Any]],
    raw_clusters: list[list[int]],
    similarity_matrix,
    cluster_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Args:
        tickets: tichetele Jira-like (aceeasi ordine ca matricea de similaritate).
        raw_clusters: liste de indici in `tickets`, cate una per cluster.
        similarity_matrix: matricea de similaritate cosinus, aliniata cu `tickets`.
        cluster_ids: id-ul fiecarui cluster, in aceeasi ordine cu `raw_clusters`.

    Returns:
        Lista de dict-uri (index, key, created, service, summary, nearest_cluster_id, nearest_similarity),
        sortata descrescator dupa similaritate (cele mai apropiate de un cluster primele).
        nearest_* sunt None cand nu exista clustere.
    """
    clustered = {i for indices in raw_clusters for i in indices}
    result: list[dict[str, Any]] = []

    for i, ticket in enumerate(tickets):
        if i in clustered:
            continue

        nearest_id, nearest_sim = None, None
        for cluster_id, indices in zip(cluster_ids, raw_clusters):
            for j in indices:
                sim = float(similarity_matrix[i][j])
                if nearest_sim is None or sim > nearest_sim:
                    nearest_id, nearest_sim = cluster_id, sim

        fields = ticket["fields"]
        result.append({
            "index": i,
            "key": ticket["key"],
            "created": fields["created"],
            "service": _service(ticket),
            "summary": fields.get("summary", ""),
            "nearest_cluster_id": nearest_id,
            "nearest_similarity": nearest_sim,
        })

    result.sort(key=lambda r: (r["nearest_similarity"] is None, -(r["nearest_similarity"] or 0.0)))
    return result