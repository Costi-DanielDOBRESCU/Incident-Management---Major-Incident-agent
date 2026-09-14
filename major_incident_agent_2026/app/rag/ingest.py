"""
Ingestion knowledge base (Etapa 4 - RAG) in cele 3 colectii ChromaDB.

Citeste cele 3 fisiere JSON din app/data/knowledge_base/ (generate de
app/models/generate_mock_data.py) si le incarca in colectiile corespunzatoare,
folosind KnowledgeBaseDocument (app.models.schemas) pentru validare.

Maparea type -> colectie (Documentatie_RO.md, sectiunea 4.3):
  post_mortem            -> historical_major_incidents
  runbook                -> runbooks
  communication_template -> communication_templates

Rulare: python -m app.rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import KnowledgeBaseDocument
from app.rag.chroma_client import get_or_create_collection

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "app" / "data" / "knowledge_base"

TYPE_TO_COLLECTION = {
    "post_mortem": "historical_major_incidents",
    "runbook": "runbooks",
    "communication_template": "communication_templates",
}

KB_FILES = {
    "post_mortem": KB_DIR / "historical_major_incidents.json",
    "runbook": KB_DIR / "runbooks.json",
    "communication_template": KB_DIR / "communication_templates.json",
}


def _load_documents(doc_type: str) -> list[KnowledgeBaseDocument]:
    path = KB_FILES[doc_type]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeBaseDocument(**item) for item in raw]


def _to_metadata(doc: KnowledgeBaseDocument) -> dict:
    """
    Converteste campurile KnowledgeBaseDocument in metadata compatibila
    ChromaDB (doar str/int/float/bool, fara liste, fara None).
    """
    metadata: dict = {
        "type": doc.type,
        "service": doc.service,
        "summary": doc.summary,
        "tags": ", ".join(doc.tags),
    }
    if doc.audience is not None:
        metadata["audience"] = doc.audience
    return metadata


def ingest_all() -> dict[str, int]:
    """
    Ingereaza toate cele 3 categorii de documente. Returneaza un dict
    {nume_colectie: nr_documente_ingerate}, util pentru verificare rapida.
    """
    counts: dict[str, int] = {}

    for doc_type, collection_name in TYPE_TO_COLLECTION.items():
        documents = _load_documents(doc_type)
        collection = get_or_create_collection(collection_name)

        ids = [doc.doc_id for doc in documents]
        texts = [f"{doc.summary}. {doc.content}" for doc in documents]
        metadatas = [_to_metadata(doc) for doc in documents]

        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        counts[collection_name] = collection.count()

    return counts


if __name__ == "__main__":
    result = ingest_all()
    print("Ingestion completa:")
    for collection_name, count in result.items():
        print(f"  - {collection_name}: {count} documente")