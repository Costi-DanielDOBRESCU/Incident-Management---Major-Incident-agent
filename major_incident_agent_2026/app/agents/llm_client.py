"""
Wrapper generic peste Ollama chat, cu output structurat (JSON Schema nativ,
suportat de Ollama 0.6.x - vezi sectiunea "format" din API).

Folosit de Assessment Agent (Etapa 5) si, mai tarziu, de Communication Agent
(Etapa 6). Doar Ollama local pentru moment (decizie curenta) - fallback pe
Groq (doc. sectiunea 9/10) e lasat pentru o etapa viitoare, cand se decide
strategia de deploy.

Principiu respectat (doc. sectiunea 4.1 / 6.1): acest modul NU valideaza cu
Pydantic - doar garanteaza JSON parsabil. Validarea Pydantic (schema de
business: IncidentAssessment, CommunicationDraft) se face in modulul
apelant (assessment_agent.py etc.), care e cel care cunoaste modelul exact.
"""

from __future__ import annotations

import json

from ollama import Client, ResponseError

from app.config import get_settings


class LlmGenerationError(Exception):
    """Ridicata cand Ollama e inaccesibil sau output-ul nu e JSON valid, dupa retry."""


def generate_structured_json(
    prompt: str,
    json_schema: dict,
    *,
    system: str | None = None,
    max_retries: int = 1,
) -> dict:
    """
    Apeleaza LLM-ul local (Ollama) cu un prompt si o schema JSON (format
    nativ Ollama), si returneaza dict-ul parsat.

    Nu valideaza campurile fata de un model Pydantic specific - doar
    garanteaza ca rezultatul e JSON sintactic valid. Validarea de business
    (range-uri, enum-uri, campuri obligatorii) se face separat, cu Pydantic,
    de catre apelant.

    Args:
        prompt: mesajul user (contine deja clusterul + contextul RAG, format
            de apelant).
        json_schema: schema JSON (dict) trimisa la Ollama prin parametrul
            "format", pentru output structurat.
        system: mesaj de sistem opțional (instructiuni de rol).
        max_retries: nr. de reincercari daca JSON-ul returnat nu se parseaza.

    Raises:
        LlmGenerationError: daca Ollama e inaccesibil sau JSON-ul nu se
            parseaza dupa toate reincercarile.
    """
    settings = get_settings()
    client = Client(host=settings.ollama_base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                model=settings.ollama_llm_model,
                messages=messages,
                format=json_schema,
            )
        except ResponseError as exc:
            raise LlmGenerationError(f"Ollama a raspuns cu eroare: {exc}") from exc
        except ConnectionError as exc:
            raise LlmGenerationError(
                f"Nu pot contacta Ollama la {settings.ollama_base_url}. "
                "Verifica daca serviciul ruleaza (`ollama serve`)."
            ) from exc

        raw_content = response["message"]["content"]

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_retries:
                # retry simplu: adaugam o instructiune explicita de corectare,
                # fara sa schimbam restul conversatiei
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {"role": "user", "content": "Raspunsul anterior nu era JSON valid. Returneaza STRICT JSON valid, fara text aditional."}
                )
                continue

    raise LlmGenerationError(
        f"LLM-ul nu a produs JSON valid dupa {max_retries + 1} incercari. Ultima eroare: {last_error}"
    )


if __name__ == "__main__":
    # sanity check rapid: `python -m app.agents.llm_client`
    test_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    }
    result = generate_structured_json(
        prompt="Say hello in one word.",
        json_schema=test_schema,
    )
    print(result)