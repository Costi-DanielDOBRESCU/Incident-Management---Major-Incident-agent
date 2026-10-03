"""
app/agents/decision_rules.py

Reguli DETERMINISTE aplicate dupa raspunsul LLM-ului in Assessment Agent.

Principiul proiectului: "LLM-ul propune, tool-urile deterministe executa". LLM-ul decide
severitatea (judecata), dar decizia care deriva direct din severitate nu trebuie lasata la
latitudinea unui model mic, care o poate contrazice (ex. SEV2 dar "not a major incident").

Regula (aceeasi cu cea din promptul Assessment Agent):
  - SEV1 / SEV2 -> is_major_incident_candidate = True, recommended_action = "propose_major_incident"
  - SEV3        -> is_major_incident_candidate = False, recommended_action in {"monitor", "dismiss"}
  - Unknown     -> nemodificat
"""

from __future__ import annotations

from typing import Any


def enforce_decision_consistency(llm_output: dict[str, Any]) -> dict[str, Any]:
    """
    Intoarce o copie a output-ului LLM cu decizia aliniata la severitate.
    Daca a trebuit corectat ceva, adauga o nota in `reasoning`, ca explicatia afisata
    utilizatorului sa nu contrazica campurile.
    """
    result = dict(llm_output)
    severity = result.get("estimated_severity")

    changed = False

    if severity in ("SEV1", "SEV2"):
        if result.get("is_major_incident_candidate") is not True:
            result["is_major_incident_candidate"] = True
            changed = True
        if result.get("recommended_action") != "propose_major_incident":
            result["recommended_action"] = "propose_major_incident"
            changed = True

    elif severity == "SEV3":
        if result.get("is_major_incident_candidate") is not False:
            result["is_major_incident_candidate"] = False
            changed = True
        if result.get("recommended_action") not in ("monitor", "dismiss"):
            result["recommended_action"] = "monitor"
            changed = True

    if changed:
        note = (
            f" [Automatic consistency check: severity {severity} implies "
            f"is_major_incident_candidate={result['is_major_incident_candidate']} and "
            f"recommended_action={result['recommended_action']}.]"
        )
        result["reasoning"] = f"{result.get('reasoning', '')}{note}"

    return result