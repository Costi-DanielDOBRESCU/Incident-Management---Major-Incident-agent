"""
Wrapper generic peste LLM-ul folosit de agenti, cu output structurat (JSON).

Provideri suportati (configurabili din .env, vezi app/config.py):
  - "ollama": LLM local, JSON Schema nativ (Ollama 0.6.x, parametrul "format").
  - "groq":   LLM in cloud (API compatibil OpenAI, response_format).

  LLM_PRIMARY_PROVIDER   = ollama | groq     (implicit: ollama)
  LLM_FALLBACK_PROVIDER  = groq | ollama     (folosit daca primary da eroare; gol = fara fallback)
  GROQ_API_KEY           = cheia din console.groq.com
  GROQ_LLM_MODEL         = ex. openai/gpt-oss-120b

Folosit de Assessment Agent (Etapa 5) si de Communication Agent (Etapa 6).

Principiu respectat (doc. sectiunea 4.1 / 6.1): acest modul NU valideaza cu
Pydantic - doar garanteaza JSON parsabil. Validarea Pydantic (schema de
business: IncidentAssessment, CommunicationDraft) se face in modulul
apelant (assessment_agent.py etc.), care e cel care cunoaste modelul exact.
"""

from __future__ import annotations

import json
from typing import Any

from ollama import Client, ResponseError

from app.config import get_settings

# Modele Groq care accepta response_format = json_schema (vezi console.groq.com/docs/structured-outputs).
# Pentru orice alt model (ex. llama-3.1-8b-instant) se foloseste json_object + schema in prompt.
GROQ_JSON_SCHEMA_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
}


_LAST_LABEL: str | None = None


def last_llm_label() -> str | None:
    """Eticheta ultimului apel LLM reusit, ex. 'openai/gpt-oss-120b (groq)'. None daca nu a existat inca."""
    return _LAST_LABEL


class LlmGenerationError(Exception):
    """Ridicata cand LLM-ul e inaccesibil sau output-ul nu e JSON valid, dupa retry."""


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _generate_ollama(
    prompt: str,
    json_schema: dict,
    system: str | None,
    max_retries: int,
    temperature: float,
) -> dict:
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
                options={"temperature": temperature},
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
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {"role": "user", "content": "Raspunsul anterior nu era JSON valid. Returneaza STRICT JSON valid, fara text aditional."}
                )
                continue

    raise LlmGenerationError(
        f"Ollama nu a produs JSON valid dupa {max_retries + 1} incercari. Ultima eroare: {last_error}"
    )


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
def _make_groq_client(api_key: str) -> Any:
    """Creat intr-o functie separata, ca pachetul `groq` sa fie necesar doar cand e folosit (si ca testele sa-l poata inlocui)."""
    from groq import Groq

    return Groq(api_key=api_key)


def _generate_groq(
    prompt: str,
    json_schema: dict,
    system: str | None,
    max_retries: int,
    temperature: float,
) -> dict:
    settings = get_settings()

    if not settings.groq_api_key:
        raise LlmGenerationError("GROQ_API_KEY lipseste. Adauga-l in .env (cheie din console.groq.com).")

    try:
        from groq import GroqError
    except ImportError as exc:
        raise LlmGenerationError("Pachetul `groq` nu e instalat. Ruleaza: pip install groq") from exc

    client = _make_groq_client(settings.groq_api_key)
    model = settings.groq_llm_model

    if model in GROQ_JSON_SCHEMA_MODELS:
        response_format: dict = {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "strict": False, "schema": json_schema},
        }
        system_text = system
    else:
        # Fara suport nativ pentru schema: cerem JSON valid si descriem schema in mesajul de sistem.
        response_format = {"type": "json_object"}
        schema_hint = f"Respond with a single JSON object that matches this JSON schema:\n{json.dumps(json_schema)}"
        system_text = f"{system}\n\n{schema_hint}" if system else schema_hint

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
        except GroqError as exc:
            raise LlmGenerationError(f"Groq a raspuns cu eroare: {exc}") from exc

        raw_content = response.choices[0].message.content or ""

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {"role": "user", "content": "Raspunsul anterior nu era JSON valid. Returneaza STRICT JSON valid, fara text aditional."}
                )
                continue

    raise LlmGenerationError(
        f"Groq nu a produs JSON valid dupa {max_retries + 1} incercari. Ultima eroare: {last_error}"
    )


_PROVIDERS = {"ollama": _generate_ollama, "groq": _generate_groq}


# ---------------------------------------------------------------------------
# API public
# ---------------------------------------------------------------------------
def generate_structured_json(
    prompt: str,
    json_schema: dict,
    *,
    system: str | None = None,
    max_retries: int = 1,
    temperature: float = 0.7,
) -> dict:
    """
    Apeleaza LLM-ul configurat (provider primar, apoi fallback daca primarul esueaza)
    cu un prompt si o schema JSON, si returneaza dict-ul parsat.

    Nu valideaza campurile fata de un model Pydantic specific - doar
    garanteaza ca rezultatul e JSON sintactic valid. Validarea de business
    (range-uri, enum-uri, campuri obligatorii) se face separat, cu Pydantic,
    de catre apelant.

    Raises:
        LlmGenerationError: daca toti providerii configurati esueaza. Mesajul contine
            eroarea fiecaruia.
    """
    settings = get_settings()

    providers = [settings.llm_primary_provider]
    fallback = (settings.llm_fallback_provider or "").strip()
    if fallback and fallback not in providers:
        providers.append(fallback)

    errors: list[str] = []
    for name in providers:
        generate = _PROVIDERS.get(name)
        if generate is None:
            errors.append(f"{name}: provider necunoscut (folosește 'ollama' sau 'groq')")
            continue
        try:
            result = generate(prompt, json_schema, system, max_retries, temperature)
        except LlmGenerationError as exc:
            errors.append(f"{name}: {exc}")
            continue

        global _LAST_LABEL
        model = getattr(settings, "groq_llm_model" if name == "groq" else "ollama_llm_model", None) or "unknown"
        _LAST_LABEL = f"{model} ({name})"
        return result

    raise LlmGenerationError(" | ".join(errors))


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