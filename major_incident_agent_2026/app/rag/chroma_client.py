"""
Client ChromaDB centralizat pentru MIA (Etapa 4 - RAG).

Foloseste embedding function-ul custom, bazat pe BGE-M3 via Ollama (aceeasi
functie folosita si de Detection Pipeline - app.detection.embeddings.create_embedding),
astfel incat toate colectiile RAG sa foloseasca exact acelasi model semantic
ca detectorul de incidente. Nu se foloseste embedding function-ul default din
ChromaDB (all-MiniLM via sentence-transformers) - ar introduce un al doilea
model semantic in proiect, in contradictie cu decizia fixata (BGE-M3).

Referinta: Documentatie_RO.md, sectiunea 4.3 (tool query_knowledge_base) si
sectiunea 5 (RAG cu ChromaDB).
"""

from __future__ import annotations

import chromadb
from chromadb import Collection
from chromadb.api.types import Documents, Embeddings

from app.config import get_settings
from app.detection.embeddings import create_embedding


class BgeM3EmbeddingFunction:
    """
    Wrapper peste create_embedding() (Ollama / BGE-M3), in formatul cerut de
    ChromaDB pentru embedding function-uri custom.
    """

    def __call__(self, input: Documents) -> Embeddings:
        return [create_embedding(text) for text in input]

    def name(self) -> str:
        # ChromaDB salveaza acest nume in metadata colectiei si il verifica
        # la reincarcare, ca sa nu mixezi din greseala doua modele diferite
        # in aceeasi colectie persistenta.
        return "bge_m3_ollama"


_client: chromadb.ClientAPI | None = None
_embedding_function = BgeM3EmbeddingFunction()


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returneaza un client ChromaDB persistent local (singleton la nivel de
    proces), scriind pe disc in settings.chroma_persist_dir.

    Nota: settings.chroma_host / settings.chroma_port sunt rezervate pentru
    o varianta viitoare cu ChromaDB ca server separat (Docker, Etapa 13).
    Nu sunt folosite inca - in Etapa 4 lucram cu PersistentClient local.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_or_create_collection(name: str) -> Collection:
    """
    Returneaza colectia ChromaDB cu numele dat, creand-o daca nu exista,
    configurata cu embedding function-ul BGE-M3 (Ollama).
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_function,
    )


if __name__ == "__main__":
    # sanity check rapid: `python -m app.rag.chroma_client`
    collection = get_or_create_collection("historical_major_incidents")
    print(f"Colectie OK: {collection.name}, documente existente: {collection.count()}")