"""
app/execution/incident_memory.py

Memorie de incidente (RAG): decizii UMANE asupra clusterelor, salvate intr-o colectie ChromaDB separata
(`incident_memory`), pe care Assessment Agent o poate interoga pentru context.

Reguli de design (de ce e separata de knowledge base-ul curatoriat):
  * Se salveaza DOAR dupa o decizie umana (declarat sau respins). Output-ul LLM nevalidat nu intra
    niciodata, ca modelul sa nu se citeze pe sine.
  * Un incident declarat NU este un post-mortem: cauza reala se cunoaste abia dupa rezolvare. Intrarea
    poarta `post_mortem = "pending"`, iar textul spune clar ca e o decizie, nu o cauza confirmata.
  * Rezultatele sunt context orientativ, cu scor minim de similaritate. Colectiile curatoriate
    (post-mortemuri, runbook-uri, sabloane) raman scrise doar de oameni.

Colectia NU face parte din ALL_COLLECTIONS (knowledge base), deci `python -m app.rag.ingestion` nu o atinge.

CLI:
    python -m app.execution.incident_memory            # listeaza intrarile
    python -m app.execution.incident_memory --reset    # goleste memoria (demo/teste)
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

COLLECTION_MEMORY = "incident_memory"

# Sub acest scor cosinus, o intrare din memorie nu e suficient de apropiata ca sa fie folosita.
MIN_SCORE = 0.55
MAX_SUMMARIES_IN_TEXT = 5
MAX_REASON_CHARS = 300


def _get_collection() -> Any:
    # import intarziat: modulul poate fi importat (si testat) fara ChromaDB / Ollama
    from app.rag.chroma_client import get_or_create_collection

    return get_or_create_collection(COLLECTION_MEMORY)


def memory_doc_id(incident_id: str) -> str:
    return f"MEM-{incident_id}"


def build_memory_text(decision: dict[str, Any]) -> str:
    """
    Textul indexat pentru o decizie. Incepe ca interogarea Assessment Agent ("<serviciu>: <rezumate>"),
    ca potrivirea semantica sa fie directa.
    """
    summaries = list(dict.fromkeys(decision.get("summaries") or []))[:MAX_SUMMARIES_IN_TEXT]
    sample = "; ".join(summaries) if summaries else decision.get("service") or "unknown"
    service = decision.get("service") or "Unknown"

    if decision["outcome"] == "declared":
        verdict = f"A human declared this cluster a major incident ({decision.get('severity') or 'Unknown'})."
    else:
        verdict = "A human REJECTED this cluster as NOT a major incident."

    reason = (decision.get("reason") or "").strip()
    reason_text = f" Reason: {reason[:MAX_REASON_CHARS]}" if reason else " No reason was recorded."

    return (
        f"{service}: {sample}. {verdict}{reason_text} "
        f"(Decision, not a confirmed root cause; no post-mortem yet.)"
    )


def remember_decision(decision: dict[str, Any], collection: Any = None) -> str:
    """Salveaza (upsert) o decizie umana in memorie. Intoarce doc_id-ul (MEM-<incident_id>)."""
    if decision.get("outcome") not in ("declared", "rejected"):
        raise ValueError("Doar deciziile umane (declared/rejected) se salveaza in memorie.")

    collection = collection or _get_collection()
    doc_id = memory_doc_id(decision["incident_id"])

    metadata = {
        "doc_id": doc_id,
        "type": "incident_memory",
        "service": decision.get("service") or "Unknown",
        "summary": f"{decision['outcome']} {decision.get('severity') or ''} {decision.get('service') or ''}".strip(),
        "outcome": decision["outcome"],
        "severity": decision.get("severity") or "Unknown",
        "incident_id": decision["incident_id"],
        "reason": (decision.get("reason") or "")[:MAX_REASON_CHARS],
        "decided_by": decision.get("decided_by") or "",
        "decided_at": decision.get("decided_at") or "",
        "human_validated": True,
        "post_mortem": "pending",
    }
    collection.upsert(ids=[doc_id], documents=[build_memory_text(decision)], metadatas=[metadata])
    return doc_id


def search_memory(
    query: str,
    service: Optional[str] = None,
    n_results: int = 2,
    min_score: float = MIN_SCORE,
    collection: Any = None,
) -> list[dict[str, Any]]:
    """
    Cauta decizii similare din trecut (acelasi serviciu, daca e dat). Intoarce doar intrarile cu
    scor >= min_score, cele mai apropiate primele. Lista goala daca memoria e goala sau indisponibila.
    """
    if not query.strip():
        return []

    try:
        collection = collection or _get_collection()
        if collection.count() == 0:
            return []
        raw = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"service": service} if service else None,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:  # noqa: BLE001 - memoria e un ajutor optional; lipsa ei nu trebuie sa opreasca evaluarea
        return []

    results = []
    for doc_id, content, metadata, distance in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        score = round(1.0 - distance, 4)
        if score < min_score:
            continue
        results.append(
            {
                "doc_id": doc_id,
                "content": content,
                "service": metadata.get("service", ""),
                "outcome": metadata.get("outcome", ""),
                "severity": metadata.get("severity", ""),
                "reason": metadata.get("reason", ""),
                "decided_at": metadata.get("decided_at", ""),
                "score": score,
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description="Vizualizeaza sau goleste memoria de incidente.")
    ap.add_argument("--reset", action="store_true", help="sterge TOATE intrarile din memorie")
    args = ap.parse_args()

    from app.rag.chroma_client import get_chroma_client

    if args.reset:
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_MEMORY)
            print("Memoria de incidente a fost golita.")
        except Exception:  # noqa: BLE001
            print("Memoria era deja goala.")
        return

    collection = _get_collection()
    raw = collection.get(include=["documents", "metadatas"])
    print(f"{collection.count()} intrari in {COLLECTION_MEMORY}\n")
    for doc_id, content, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        print(f"{doc_id}  [{metadata.get('outcome')}, {metadata.get('severity')}, {metadata.get('service')}]  "
              f"{metadata.get('decided_at')} by {metadata.get('decided_by')}")
        print(f"    {content}\n")


if __name__ == "__main__":
    _main()