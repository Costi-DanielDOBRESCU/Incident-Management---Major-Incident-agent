"""Embedding generation using the local Ollama embedding model."""

from ollama import Client

from app.config import get_settings


def create_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text using Ollama.
    """
    if not text.strip():
        raise ValueError("text must not be empty.")

    settings = get_settings()
    client = Client(host=settings.ollama_base_url)

    response = client.embed(
        model=settings.ollama_embed_model,
        input=text,
    )

    embeddings = response["embeddings"]

    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")

    return embeddings[0]