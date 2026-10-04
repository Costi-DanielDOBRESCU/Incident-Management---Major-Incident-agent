"""Tests for the LLM client: provider dispatch, fallback and Groq response formats (no network)."""

import json
from types import SimpleNamespace

import pytest

from app.agents import llm_client
from app.agents.llm_client import LlmGenerationError, generate_structured_json

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def _settings(**overrides):
    base = dict(
        llm_primary_provider="ollama",
        llm_fallback_provider="",
        groq_api_key="test-key",
        groq_llm_model="openai/gpt-oss-120b",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeGroq:
    """Imită clientul Groq: înregistrează apelurile și întoarce răspunsurile date."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def use_settings(monkeypatch):
    def _apply(**overrides):
        monkeypatch.setattr(llm_client, "get_settings", lambda: _settings(**overrides))

    return _apply


# ---------- dispatch + fallback ----------

def test_primary_ollama_is_used_when_it_works(monkeypatch, use_settings):
    use_settings()
    monkeypatch.setitem(llm_client._PROVIDERS, "ollama", lambda *a: {"answer": "ollama"})
    monkeypatch.setitem(llm_client._PROVIDERS, "groq", lambda *a: pytest.fail("groq must not be called"))

    assert generate_structured_json("p", SCHEMA) == {"answer": "ollama"}


def test_fallback_to_groq_when_primary_fails(monkeypatch, use_settings):
    use_settings(llm_fallback_provider="groq")

    def failing(*a):
        raise LlmGenerationError("ollama down")

    monkeypatch.setitem(llm_client._PROVIDERS, "ollama", failing)
    monkeypatch.setitem(llm_client._PROVIDERS, "groq", lambda *a: {"answer": "groq"})

    assert generate_structured_json("p", SCHEMA) == {"answer": "groq"}


def test_error_lists_every_provider_failure(monkeypatch, use_settings):
    use_settings(llm_fallback_provider="groq")

    def fail_ollama(*a):
        raise LlmGenerationError("ollama down")

    def fail_groq(*a):
        raise LlmGenerationError("groq down")

    monkeypatch.setitem(llm_client._PROVIDERS, "ollama", fail_ollama)
    monkeypatch.setitem(llm_client._PROVIDERS, "groq", fail_groq)

    with pytest.raises(LlmGenerationError) as exc_info:
        generate_structured_json("p", SCHEMA)

    assert "ollama: ollama down" in str(exc_info.value)
    assert "groq: groq down" in str(exc_info.value)


def test_no_fallback_when_empty(monkeypatch, use_settings):
    use_settings(llm_fallback_provider="")

    def failing(*a):
        raise LlmGenerationError("ollama down")

    monkeypatch.setitem(llm_client._PROVIDERS, "ollama", failing)
    monkeypatch.setitem(llm_client._PROVIDERS, "groq", lambda *a: pytest.fail("groq must not be called"))

    with pytest.raises(LlmGenerationError):
        generate_structured_json("p", SCHEMA)


def test_unknown_provider_is_reported(use_settings):
    use_settings(llm_primary_provider="nope")

    with pytest.raises(LlmGenerationError, match="provider necunoscut"):
        generate_structured_json("p", SCHEMA)


# ---------- Groq ----------

def test_groq_requires_api_key(use_settings):
    use_settings(llm_primary_provider="groq", groq_api_key="")

    with pytest.raises(LlmGenerationError, match="GROQ_API_KEY"):
        generate_structured_json("p", SCHEMA)


def test_groq_uses_json_schema_for_supported_model(monkeypatch, use_settings):
    use_settings(llm_primary_provider="groq", groq_llm_model="openai/gpt-oss-120b")
    fake = FakeGroq([json.dumps({"answer": "hi"})])
    monkeypatch.setattr(llm_client, "_make_groq_client", lambda key: fake)

    result = generate_structured_json("prompt", SCHEMA, system="sys", temperature=0.1)

    assert result == {"answer": "hi"}
    call = fake.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["temperature"] == 0.1
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"] == SCHEMA
    assert call["messages"][0] == {"role": "system", "content": "sys"}


def test_groq_uses_json_object_and_schema_hint_for_other_models(monkeypatch, use_settings):
    use_settings(llm_primary_provider="groq", groq_llm_model="llama-3.1-8b-instant")
    fake = FakeGroq([json.dumps({"answer": "hi"})])
    monkeypatch.setattr(llm_client, "_make_groq_client", lambda key: fake)

    generate_structured_json("prompt", SCHEMA, system="sys")

    call = fake.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    system_message = call["messages"][0]["content"]
    assert system_message.startswith("sys")
    assert '"answer"' in system_message  # schema inclusa in prompt


def test_groq_retries_once_on_invalid_json(monkeypatch, use_settings):
    use_settings(llm_primary_provider="groq")
    fake = FakeGroq(["not json", json.dumps({"answer": "ok"})])
    monkeypatch.setattr(llm_client, "_make_groq_client", lambda key: fake)

    assert generate_structured_json("p", SCHEMA, max_retries=1) == {"answer": "ok"}
    assert len(fake.calls) == 2
    assert "nu era JSON valid" in fake.calls[1]["messages"][-1]["content"]


def test_groq_gives_up_after_retries(monkeypatch, use_settings):
    use_settings(llm_primary_provider="groq")
    fake = FakeGroq(["bad", "still bad"])
    monkeypatch.setattr(llm_client, "_make_groq_client", lambda key: fake)

    with pytest.raises(LlmGenerationError, match="Groq nu a produs JSON valid"):
        generate_structured_json("p", SCHEMA, max_retries=1)


def test_groq_api_error_becomes_llm_generation_error(monkeypatch, use_settings):
    from groq import GroqError

    use_settings(llm_primary_provider="groq")
    fake = FakeGroq([GroqError("rate limited")])
    monkeypatch.setattr(llm_client, "_make_groq_client", lambda key: fake)

    with pytest.raises(LlmGenerationError, match="Groq a raspuns cu eroare"):
        generate_structured_json("p", SCHEMA)