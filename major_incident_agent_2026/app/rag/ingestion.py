"""
Ingestion knowledge base -> ChromaDB (Etapa 4 - RAG).

Citeste cele 3 fisiere JSON din app/data/knowledge_base/
(historical_major_incidents.json, runbooks.json, communication_templates.json)
si populeaza colectiile ChromaDB corespunzatoare (app.rag.chroma_client).

Text indexat: campul "content" (descriptiv, spre diferenta de "summary" care
e doar titlu) - simetric cu Detection Pipeline, care indexeaza summary+description,
nu doar summary (doc. sectiunea 10, risc "calitate slaba embeddings pe text scurt").

Metadata pastrata per document: doc_id, type, service, summary, tags (join
cu virgula - ChromaDB nu accepta liste in metadata), plus "audience" pentru
communication_templates. rag_sources (in Assessment/CommunicationDraft) se
populeaza direct din doc_id-urile returnate la query.

Referinta: Documentatie_RO.md sectiunea 5.1 (KnowledgeBaseDocument) si 5.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.rag.chroma_client import (
    COLLECTION_HISTORICAL,
    COLLECTION_RUNBOOKS,
    COLLECTION_TEMPLATES,
    get_or_create_collection,
)

# Nume fisier -> nume colectie ChromaDB. Un singur loc unde se face maparea,
# ca sa nu se repete stringuri hardcodate in alte module.
_FILE_TO_COLLECTION = {
    "historical_major_incidents.json": COLLECTION_HISTORICAL,
    "runbooks.json": COLLECTION_RUNBOOKS,
    "communication_templates.json": COLLECTION_TEMPLATES,
}


def _kb_dir() -> Path:
    settings = get_settings()
    # chroma_persist_dir e "./app/data/chroma_store" -> knowledge_base e vecin,
    # in "./app/data/knowledge_base". Derivam din acelasi settings, nu hardcodam
    # doua cai independente care se pot dezalinia.
    return Path(settings.chroma_persist_dir).parent / "knowledge_base"


def _load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list):
        raise ValueError(f"{path} trebuie sa contina un array JSON la nivel de root.")
    return docs


def _build_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "type": doc["type"],
        "service": doc["service"],
        "summary": doc["summary"],
    }
    tags = doc.get("tags")
    if tags:
        metadata["tags"] = ",".join(tags)
    audience = doc.get("audience")
    if audience:
        metadata["audience"] = audience
    return metadata


def ingest_file(path: Path, collection_name: str, *, verbose: bool = True) -> int:
    """
    Ingera un singur fisier JSON in colectia ChromaDB indicata.
    Foloseste upsert (nu add) - rerularea ingestion-ului e idempotenta,
    nu produce duplicate la a doua rulare pe acelasi doc_id.
    """
    docs = _load_documents(path)
    collection = get_or_create_collection(collection_name)

    ids = [doc["doc_id"] for doc in docs]
    texts = [doc["content"] for doc in docs]
    metadatas = [_build_metadata(doc) for doc in docs]

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    if verbose:
        print(f"  {path.name:<35} -> {collection_name:<28} ({len(docs)} documente)")

    return len(docs)


def ingest_all(*, verbose: bool = True) -> dict[str, int]:
    """
    Ruleaza ingestion pentru toate cele 3 fisiere din knowledge_base/.
    Returneaza un dict {nume_fisier: nr_documente_ingerate}.
    """
    kb_dir = _kb_dir()
    results: dict[str, int] = {}

    for filename, collection_name in _FILE_TO_COLLECTION.items():
        file_path = kb_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Nu gasesc {file_path} - verifica structura app/data/knowledge_base/.")
        results[filename] = ingest_file(file_path, collection_name, verbose=verbose)

    return results


if __name__ == "__main__":
    # rulare: `python -m app.rag.ingestion`
    print(f"Knowledge base dir: {_kb_dir()}")
    total = ingest_all()
    print(f"\nTotal documente ingerate: {sum(total.values())} (asteptat: 18)")