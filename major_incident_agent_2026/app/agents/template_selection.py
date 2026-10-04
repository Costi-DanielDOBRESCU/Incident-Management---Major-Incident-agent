"""
app/agents/template_selection.py

Selectie DETERMINISTA a sabloanelor de comunicare pentru Communication Agent.

De ce nu mai folosim cautarea semantica (top-2) pe toata colectia: colectia contine acum sabloane
pentru mai multe etape (declarare, update, rezolvat, "nu e incident major", incident la furnizor)
si pentru mai multe servicii. O cautare semantica poate alege un sablon pentru alta etapa sau alt
serviciu (ex. un sablon de retea pentru un incident de portal, sau cel pentru incident la furnizor
cand cauza e interna). Comunicarea la declarare are nevoie de un sablon potrivit etapei, severitatii
si serviciului, deci alegerea se face prin reguli, iar continutul se citeste din ChromaDB dupa doc_id.

Reguli (etapa: incident tocmai declarat):
  end_users:   SEV1 -> TPL-COMM-USER-004, altfel TPL-COMM-USER-001
               + sablonul specific serviciului, daca exista (Cloud, Network, Email, Print)
  management:  SEV1 -> TPL-COMM-MGMT-004, altfel TPL-COMM-MGMT-001
"""

from __future__ import annotations

from typing import Any, Sequence

USER_BASE = "TPL-COMM-USER-001"
USER_SEV1 = "TPL-COMM-USER-004"
MGMT_BASE = "TPL-COMM-MGMT-001"
MGMT_SEV1 = "TPL-COMM-MGMT-004"

# Sabloane specifice serviciului, scrise independent de cauza (valabile pentru orice incident al serviciului)
SERVICE_SPECIFIC_USER_TEMPLATES: dict[str, str] = {
    "Cloud Storage": "TPL-COMM-USER-005",
    "Network/Switch": "TPL-COMM-USER-006",
    "Email/Exchange": "TPL-COMM-USER-007",
    "Print Services": "TPL-COMM-USER-008",
}


def select_template_ids(audience: str, severity: str, service: str) -> list[str]:
    """Intoarce doc_id-urile sabloanelor de folosit, in ordinea in care apar in prompt."""
    if audience == "end_users":
        ids = [USER_SEV1 if severity == "SEV1" else USER_BASE]
        specific = SERVICE_SPECIFIC_USER_TEMPLATES.get(service)
        if specific:
            ids.append(specific)
        return ids

    if audience == "management":
        return [MGMT_SEV1 if severity == "SEV1" else MGMT_BASE]

    raise ValueError(f"Audienta necunoscuta: {audience!r}")


def fetch_templates(doc_ids: Sequence[str], collection: Any = None) -> list[dict[str, Any]]:
    """
    Citeste sabloanele dupa doc_id din colectia ChromaDB `communication_templates`,
    pastrand ordinea ceruta. Forma rezultatului e compatibila cu query_knowledge_base
    (doc_id, summary, content, service, score); `score` e 1.0 pentru ca selectia nu e semantica.
    Sabloanele inexistente sunt omise.
    """
    if collection is None:
        from app.rag.chroma_client import COLLECTION_TEMPLATES, get_or_create_collection

        collection = get_or_create_collection(COLLECTION_TEMPLATES)

    raw = collection.get(ids=list(doc_ids), include=["documents", "metadatas"])
    found = {
        doc_id: (content, metadata or {})
        for doc_id, content, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"])
    }

    results: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        if doc_id not in found:
            continue
        content, metadata = found[doc_id]
        results.append(
            {
                "doc_id": doc_id,
                "summary": metadata.get("summary", ""),
                "content": content,
                "service": metadata.get("service", ""),
                "score": 1.0,
            }
        )
    return results