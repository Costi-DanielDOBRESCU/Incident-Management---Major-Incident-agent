"""
Client ChromaDB centralizat pentru MIA (Etapa 4 - RAG).

Foloseste un embedding function custom bazat pe BGE-M3 via Ollama (aceeasi
functie folosita si de Detection Pipeline - app.detection.embeddings.create_embedding),
astfel incat toate colectiile RAG sa foloseasca exact acelasi model semantic
ca detectorul de incidente. Nu se foloseste embedding function-ul default din
ChromaDB (all-MiniLM via sentence-transformers) - ar introduce un al doilea
model semantic in proiect, in contradictie cu decizia fixata (BGE-M3).

Referinta: Documentatie_RO.md, sectiunea 4.3 (tool query_knowledge_base),
sectiunea 5 (RAG cu ChromaDB) si sectiunea 6.5 (praguri - cosine).
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from app.config import get_settings
from app.detection.embeddings import create_embedding

# Numele celor 3 colectii RAG (doc. sectiunea 5.3). Importate de toate
# modulele care fac ingestion sau query - nu se scriu string-uri hardcodate.
COLLECTION_HISTORICAL = "historical_major_incidents"
COLLECTION_RUNBOOKS = "runbooks"
COLLECTION_TEMPLATES = "communication_templates"

ALL_COLLECTIONS = (
    COLLECTION_HISTORICAL,
    COLLECTION_RUNBOOKS,
    COLLECTION_TEMPLATES,
)


@register_embedding_function
class BgeM3EmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Wrapper peste create_embedding() (Ollama / BGE-M3), in formatul cerut de
    ChromaDB 1.x pentru embedding function-uri custom.

    Nota de performanta: se apeleaza Ollama o data per document. Pentru cele
    18 documente ale knowledge base-ului si pentru query-uri individuale este
    complet acceptabil. Batching-ul real (un singur apel cu lista de texte)
    ramane o optimizare pentru mai tarziu, daca volumul creste.
    """
    def __init__(self) -> None:
        # Explicit, chiar daca nu avem parametri: ChromaDB 1.x cere __init__
        # definit pe embedding function-urile custom (altfel DeprecationWarning,
        # iar in versiuni viitoare eroare). Configurarea modelului vine din
        # app.config (.env), nu din argumente aici.
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        return [create_embedding(text) for text in input]

    @staticmethod
    def name() -> str:
        # Numele este salvat in metadata colectiei persistente si verificat la
        # reincarcare, ca sa nu se amestece doua modele diferite in aceeasi colectie.
        return "bge_m3_ollama"

    def get_config(self) -> dict[str, Any]:
        # Configuratia e citita din app.config (.env), nu din Chroma, deci
        # nu avem parametri de serializat aici.
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "BgeM3EmbeddingFunction":
        return BgeM3EmbeddingFunction()


_client: chromadb.ClientAPI | None = None
_embedding_function = BgeM3EmbeddingFunction()


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returneaza un client ChromaDB persistent local (singleton la nivel de
    proces), scriind pe disc in settings.chroma_persist_dir.

    Nota: settings.chroma_host / settings.chroma_port sunt rezervate pentru
    varianta cu ChromaDB ca server separat (Docker, Etapa 13). Nu sunt
    folosite inca - in Etapa 4 lucram cu PersistentClient local.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_or_create_collection(name: str) -> Collection:
    """
    Returneaza colectia ChromaDB cu numele dat, creand-o daca nu exista,
    configurata cu embedding function-ul BGE-M3 (Ollama) si cu spatiul de
    distanta COSINE.

    Cosine este obligatoriu: intreg proiectul este calibrat pe similaritate
    cosinus (prag 0.75, doc. sectiunea 6.5). Default-ul ChromaDB este L2,
    ceea ce ar produce scoruri incomparabile cu cele din Detection Pipeline.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_function,
        configuration={"hnsw": {"space": "cosine"}},
    )


if __name__ == "__main__":
    # sanity check rapid: `python -m app.rag.chroma_client`
    settings = get_settings()
    print(f"Persist dir : {settings.chroma_persist_dir}")
    print(f"Embed model : {settings.ollama_embed_model}")

    for collection_name in ALL_COLLECTIONS:
        collection = get_or_create_collection(collection_name)
        print(f"  {collection.name:<30} documente: {collection.count()}")