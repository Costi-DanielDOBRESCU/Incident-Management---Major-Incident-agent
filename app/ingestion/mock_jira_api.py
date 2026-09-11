"""
app/ingestion/mock_jira_api.py

Mock API Jira-like (Etapa 1-2, fișier 3/4).

Vezi Documentatie_RO.md:
  - secțiunea 5.4 — formatul mock API (endpoint /rest/api/2/search)
  - secțiunea 4.3 — tool-ul determinist `fetch_tickets(since, limit)`

Rulare manuală (opțional, pentru testare cu curl/browser):
    uvicorn app.ingestion.mock_jira_api:app --reload --port 8001
    GET http://127.0.0.1:8001/rest/api/2/search?since=2026-08-19T09:00:00+0000&limit=50
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Config / căi
# ---------------------------------------------------------------------------

# app/ingestion/mock_jira_api.py -> app/ -> major_incident_agent/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TICKETS_FILE = BASE_DIR / "app" / "data" / "mock_tickets" / "tickets.json"

app = FastAPI(
    title="Mock Jira API",
    description=(
        "Simulează endpoint-ul /rest/api/2/search din Jira REST API, "
        "pentru testarea locală a MIA fără o instanță Jira reală."
    ),
    version="0.1.0",
)


def _load_issues() -> list[dict[str, Any]]:
    """Încarcă tichetele mock din disc (generate de scripts/generate_mock_data.py)."""
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(
            f"Nu găsesc {TICKETS_FILE}. Rulează întâi:\n"
            f"    python -m scripts.generate_mock_data"
        )
    with open(TICKETS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["issues"]


def _parse_jira_datetime(value: str) -> datetime:
    """
    Parsează un timestamp în formatul folosit de câmpul `created`, ex:
        "2026-08-01T09:24:00.000+0000"
    Acceptă și variante fără milisecunde sau cu 'Z' în loc de offset numeric.
    """
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+0000"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    raise ValueError(f"Format de dată necunoscut pentru mock Jira API: {value!r}")


# ---------------------------------------------------------------------------
# Endpoint principal
# ---------------------------------------------------------------------------

@app.get("/rest/api/2/search")
def search_issues(
    since: Optional[str] = Query(
        default=None,
        description=(
            "Timestamp ISO (ex: 2026-08-19T09:00:00+0000). Returnează doar "
            "tichetele create la sau după acest moment."
        ),
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Nr. maxim de tichete în pagina returnată."),
) -> dict[str, Any]:
    """
    Echivalentul mock al `GET /rest/api/2/search?since=...&limit=...` din Jira.

    La fel ca în Jira real: `total` = numărul TOTAL de tichete care se
    potrivesc filtrului (indiferent de `limit`), iar `issues` = pagina
    efectiv returnată, limitată la `limit` elemente, sortată cronologic.
    """
    issues = _load_issues()

    if since is not None:
        since_dt = _parse_jira_datetime(since)
        issues = [
            issue
            for issue in issues
            if _parse_jira_datetime(issue["fields"]["created"]) >= since_dt
        ]

    total = len(issues)

    issues = sorted(issues, key=lambda i: _parse_jira_datetime(i["fields"]["created"]))
    page = issues[:limit]

    return {"total": total, "issues": page}


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check simplu, util pentru teste/CI."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# fetch_tickets() — tool-ul determinist din secțiunea 4.3
# ---------------------------------------------------------------------------
# Wrapper peste endpoint-ul de mai sus, folosind FastAPI TestClient — evită
# să pornim un server real (uvicorn) doar pentru a apela intern acest tool
# din Orchestrator/teste. Dacă vom conecta un Jira real, doar acest wrapper
# se schimbă (rămâne aceeași semnătură `fetch_tickets(since, limit) -> dict`).

_client = TestClient(app)


def fetch_tickets(since: Optional[str] = None, limit: int = 50) -> dict[str, Any]:
    """
    Tool determinist: interoghează mock API-ul Jira-like de mai sus și
    returnează tichetele noi, în același format Jira-like ({"total", "issues"}).

    Args:
        since: timestamp ISO, ex. "2026-08-19T09:00:00+0000". Dacă e None,
            returnează cele mai vechi `limit` tichete din tot setul mock.
        limit: numărul maxim de tichete din pagina returnată (implicit 50).

    Returns:
        dict Jira-like: {"total": int, "issues": [...]}
    """
    params: dict[str, Any] = {"limit": limit}
    if since is not None:
        params["since"] = since

    response = _client.get("/rest/api/2/search", params=params)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)