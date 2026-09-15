"""
Tool query_knowledge_base(query, collection) pentru RAG (Etapa 4).

Apelat de Assessment Agent (colectii historical_major_incidents, runbooks)
si Communication Agent (colectia communication_templates) - doc. sectiunea
4.3 si 5.3. Rezultatele expun doc_id, folosit direct in campul rag_sources
din IncidentAssessment / CommunicationDraft (sectiunea 6.2).

Scor: similaritate cosine (1 - distanta cosine returnata de ChromaDB), NU
distanta bruta - pentru consistenta cu pragurile din documentatie (sectiunea
6.5), toate exprimate ca similaritate (ex. 0.75), nu ca distanta.
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.rag.chroma_client import ALL_COLLECTIONS, get_or_create_collection


class RagResult(TypedDict):
    doc_id: str
    summary: str
    content: str
    service: str
    score: float


def query_knowledge_base(
    query: str,
    collection: str,
    n_results: int = 3,
    service_filter: str | None = None,
) -> list[RagResult]:
    """
    Interogheaza o colectie ChromaDB si returneaza cele mai relevante
    documente, ca similaritate cosine descrescatoare.

    Args:
        query: textul de cautare (ex. "VPN outage communication template").
        collection: una din ALL_COLLECTIONS (historical_major_incidents,
            runbooks, communication_templates).
        n_results: nr. maxim de documente returnate.
        service_filter: daca setat, restrictioneaza cautarea la documentele
            cu acest "service" exact (ex. "VPN Gateway"). Util cand Assessment
            Agent deja cunoaste affected_service din cluster si nu are rost
            sa primeasca runbook-uri de la alte servicii.

    Returns:
        Lista de RagResult, sortata descrescator dupa score (similaritate).
        Lista goala daca colectia e vida sau filtrul nu are potriviri.
    """
    if not query.strip():
        raise ValueError("query nu poate fi gol.")

    if collection not in ALL_COLLECTIONS:
        raise ValueError(
            f"Colectie necunoscuta: '{collection}'. Valori valide: {ALL_COLLECTIONS}."
        )

    coll = get_or_create_collection(collection)

    where: dict[str, Any] | None = {"service": service_filter} if service_filter else None

    raw = coll.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results: list[RagResult] = []
    for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB, cu space="cosine", returneaza distanta cosine = 1 - similaritate.
        similarity = 1.0 - distance
        results.append(
            RagResult(
                doc_id=doc_id,
                summary=metadata.get("summary", ""),
                content=content,
                service=metadata.get("service", ""),
                score=round(similarity, 4),
            )
        )

    return results


if __name__ == "__main__":
    # sanity check rapid: `python -m app.rag.query`
    print("Test 1: historical_major_incidents, query generic VPN")
    for r in query_knowledge_base("VPN authentication outage", "historical_major_incidents"):
        print(f"  {r['doc_id']:<15} score={r['score']:.4f}  {r['summary']}")

    print("\nTest 2: runbooks, cu service_filter='VPN Gateway'")
    for r in query_knowledge_base(
        "severity criteria multiple users", "runbooks", service_filter="VPN Gateway"
    ):
        print(f"  {r['doc_id']:<15} score={r['score']:.4f}  {r['summary']}")

    print("\nTest 3: communication_templates, user-facing outage")
    for r in query_knowledge_base("service disruption in progress", "communication_templates"):
        print(f"  {r['doc_id']:<15} score={r['score']:.4f}  {r['summary']}")

    print("\nTest 4: colectie invalida (trebuie sa dea ValueError)")
    try:
        query_knowledge_base("test", "colectie_inexistenta")
    except ValueError as e:
        print(f"  OK, eroare prinsa: {e}")