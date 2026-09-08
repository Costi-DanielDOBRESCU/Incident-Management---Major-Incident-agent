"""
Configurare centralizata a aplicatiei Major Incident Agent (MIA).

Toate modulele (detection, rag, agents, orchestrator, execution) citesc
pragurile si setarile provider-ilor LLM de aici, niciodata hardcodat.
Sursa: fisierul .env (vezi .env.example pentru template).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM providers ---
    llm_primary_provider: str = "ollama"       # "ollama" | "groq"
    llm_fallback_provider: str = "groq"        # folosit daca primary da eroare/timeout

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    groq_api_key: str = ""
    groq_llm_model: str = "llama-3.1-8b-instant"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./app/data/chroma_store"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # --- Praguri human-in-the-loop / detectie (doc, sectiunea 6.5) ---
    similarity_threshold: float = 0.75
    min_tickets_per_cluster: int = 3
    assessment_confidence_threshold: float = 0.6
    clustering_window_minutes: int = 20

    # --- Observabilitate ---
    phoenix_host: str = "localhost"
    phoenix_port: int = 6006
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    # --- API / App ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    mock_jira_port: int = 8002
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Settings sunt cache-uite (singleton) - se citesc o singura data din .env."""
    return Settings()


if __name__ == "__main__":
    # sanity check rapid: `python -m app.config`
    s = get_settings()
    print(s.model_dump())
