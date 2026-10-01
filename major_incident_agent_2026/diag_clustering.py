"""
diag_clustering.py  (script de diagnostic, NU face parte din aplicatie)

Ia tichetele din tabelul local (app/data/tickets.db) si arata, pentru 3 variante de text
trimis la embeddings, matricea de similaritate si clusterele rezultate la mai multe praguri.

Rulare (din radacina proiectului, cu Ollama pornit si venv activ):
    python diag_clustering.py
"""

from __future__ import annotations

from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.ingestion import ticket_store

THRESHOLDS = [0.60, 0.65, 0.70, 0.75]
MIN_SIZE = 3


def _service(t: dict) -> str:
    comps = t["fields"].get("components") or []
    return comps[0]["name"] if comps else "Unknown"


VARIANTS = {
    "A. doar summary (varianta actuala)":
        lambda t: t["fields"]["summary"],
    "B. serviciu + summary":
        lambda t: f"{_service(t)}: {t['fields']['summary']}",
    "C. serviciu + summary + description":
        lambda t: f"Service: {_service(t)}. {t['fields']['summary']}. {t['fields'].get('description', '')}",
}


def main() -> None:
    tickets = ticket_store.list_raw_issues()
    if len(tickets) < 3:
        print("Prea putine tichete in tabel. Lasa mock-ul + ingestorul sa trimita tichete si reincearca.")
        return

    keys = [t["key"] for t in tickets]
    print(f"{len(tickets)} tichete:")
    for t in tickets:
        print(f"  {t['key']:<12} {_service(t):<18} {t['fields']['summary']}")

    for name, build_text in VARIANTS.items():
        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)

        embeddings = [create_embedding(build_text(t)) for t in tickets]
        matrix = calculate_similarity_matrix(embeddings)

        header = "          " + " ".join(f"{k[-5:]:>6}" for k in keys)
        print(header)
        for k, row in zip(keys, matrix):
            print(f"{k[-5:]:>8}  " + " ".join(f"{v:6.2f}" for v in row))

        for th in THRESHOLDS:
            clusters = cluster_similar_tickets(matrix, threshold=th, min_cluster_size=MIN_SIZE)
            shown = [[keys[i][-5:] for i in c] for c in clusters]
            print(f"  prag {th:.2f} -> {len(clusters)} cluster(e): {shown}")


if __name__ == "__main__":
    main()