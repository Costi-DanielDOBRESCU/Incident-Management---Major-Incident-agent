"""Tests for the deterministic consistency rules applied after the LLM assessment."""

from app.agents.decision_rules import enforce_decision_consistency


def _output(severity, candidate, action, reasoning="LLM reasoning."):
    return {
        "estimated_severity": severity,
        "is_major_incident_candidate": candidate,
        "recommended_action": action,
        "confidence": 0.8,
        "reasoning": reasoning,
    }


def test_sev2_contradicted_by_llm_is_corrected():
    # exact cazul observat: SEV2 dar "not a major incident candidate", action=monitor
    result = enforce_decision_consistency(_output("SEV2", False, "monitor"))

    assert result["is_major_incident_candidate"] is True
    assert result["recommended_action"] == "propose_major_incident"
    assert "Automatic consistency check" in result["reasoning"]
    assert result["reasoning"].startswith("LLM reasoning.")


def test_sev1_is_corrected():
    result = enforce_decision_consistency(_output("SEV1", False, "dismiss"))

    assert result["is_major_incident_candidate"] is True
    assert result["recommended_action"] == "propose_major_incident"


def test_sev3_proposed_as_major_is_corrected():
    result = enforce_decision_consistency(_output("SEV3", True, "propose_major_incident"))

    assert result["is_major_incident_candidate"] is False
    assert result["recommended_action"] == "monitor"


def test_consistent_output_is_left_untouched():
    for output in (
        _output("SEV2", True, "propose_major_incident"),
        _output("SEV3", False, "monitor"),
        _output("SEV3", False, "dismiss"),
    ):
        result = enforce_decision_consistency(output)

        assert result == output  # nicio modificare, nicio nota adaugata


def test_unknown_severity_is_not_modified():
    output = _output("Unknown", False, "monitor")

    assert enforce_decision_consistency(output) == output


def test_input_is_not_mutated():
    output = _output("SEV2", False, "monitor")
    snapshot = dict(output)

    enforce_decision_consistency(output)

    assert output == snapshot