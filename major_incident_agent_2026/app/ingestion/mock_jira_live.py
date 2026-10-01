"""
app/ingestion/mock_jira_live.py

Mock Jira LIVE: tichetele "sosesc" in timp real, nu sunt servite dintr-un fisier static.

- Store in memorie cu tichetele primite. `created` = momentul primirii (ca in Jira real).
- GET  /rest/api/2/search?since=&limit=  -> acelasi format ca mock-ul static ({"total", "issues"}).
- POST /rest/api/2/issue                 -> creeaza un tichet (ca in Jira real).
- POST /mock/start                      -> porneste fluxul automat de tichete (idempotent).
- POST /mock/reset                      -> opreste fluxul si goleste store-ul.
- GET  /mock/status, /health

Simulatorul emite tichete la intervale de ~TICKET_INTERVAL_SECONDS, in ordine aleatorie:
la fiecare pas alege o categorie aleatorie (dintre cele care mai au tichete), apoi un tichet
aleatoriu din ea. Niciun tichet nu e emis de doua ori pe parcursul unui flux.

mock_jira_api.py (static) ramane neschimbat - il folosesc testele existente.

Rulare (din radacina proiectului, cu venv activ):
    uvicorn app.ingestion.mock_jira_live:app --port 8002
"""

from __future__ import annotations

import asyncio
import random
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.ingestion.mock_jira_api import (
    DEMO_MAJOR_INCIDENTS,
    _load_issues,
    _parse_jira_datetime,
)

app = FastAPI(
    title="Mock Jira API (live)",
    description="Simuleaza sosirea in timp real a tichetelor, in format Jira-like.",
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# Parametri simulator (modifica aici pentru teste)
# ---------------------------------------------------------------------------
TICKET_INTERVAL_SECONDS = 5.0    # intervalul mediu dintre tichete
INTERVAL_JITTER_SECONDS = 2.0    # variatie aleatorie +/- (0 = interval fix)

# ---------------------------------------------------------------------------
# Store in memorie
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_store: list[dict[str, Any]] = []
_generation = 0          # creste la /mock/stop si /mock/reset -> replay-urile vechi se opresc
_stream_state = "idle"  # idle | running | finished
_remaining = 0
_key_counter = 90000     # chei generate pentru tichete create prin POST /issue


def _now_jira() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}+0000"


def _add_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Adauga in store un issue, cu `created` = acum. Daca cheia exista, adauga sufix -R<n>."""
    with _lock:
        existing = {i["key"] for i in _store}
        base_key = issue["key"]
        key, n = base_key, 1
        while key in existing:
            n += 1
            key = f"{base_key}-R{n}"
        issue = {**issue, "key": key, "fields": {**issue["fields"], "created": _now_jira()}}
        _store.append(issue)
        return issue


# ---------------------------------------------------------------------------
# Endpoint-uri Jira-like
# ---------------------------------------------------------------------------
@app.get("/rest/api/2/search")
def search_issues(
    since: Optional[str] = Query(default=None, description="Timestamp ISO; tichete create la/dupa acest moment."),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    with _lock:
        issues = list(_store)

    if since is not None:
        try:
            since_dt = _parse_jira_datetime(since)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        issues = [i for i in issues if _parse_jira_datetime(i["fields"]["created"]) >= since_dt]

    issues.sort(key=lambda i: _parse_jira_datetime(i["fields"]["created"]))
    return {"total": len(issues), "issues": issues[:limit]}


class NewIssueFields(BaseModel):
    summary: str
    description: str = ""
    priority: str = "P3 - Medium"
    service: str = "Unknown"
    reporter: str = "mock_user"
    location: str = ""
    status: str = "Open"


class NewIssue(BaseModel):
    fields: NewIssueFields


@app.post("/rest/api/2/issue", status_code=201)
def create_issue(body: NewIssue) -> dict[str, Any]:
    global _key_counter
    with _lock:
        _key_counter += 1
        key = f"INC-{_key_counter}"
    f = body.fields
    issue = _add_issue({
        "key": key,
        "fields": {
            "summary": f.summary,
            "description": f.description,
            "priority": {"name": f.priority},
            "status": {"name": f.status},
            "components": [{"name": f.service}],
            "reporter": {"name": f.reporter},
            "labels": [],
            "location": f.location,
        },
    })
    return {"key": issue["key"], "created": issue["fields"]["created"]}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def _build_pool() -> dict[str, list[dict[str, Any]]]:
    """Categorie -> lista de tichete (categoriile = grupurile din DEMO_MAJOR_INCIDENTS)."""
    by_key = {i["key"]: i for i in _load_issues()}
    pool: dict[str, list[dict[str, Any]]] = {}
    for category, keys in DEMO_MAJOR_INCIDENTS.items():
        issues = [by_key[k] for k in keys if k in by_key]
        if issues:
            pool[category] = issues
    return pool


async def _stream(generation: int) -> None:
    global _stream_state, _remaining
    pool = _build_pool()
    _remaining = sum(len(v) for v in pool.values())
    try:
        while pool:
            if generation != _generation:
                return
            category = random.choice(list(pool))
            issue = pool[category].pop(random.randrange(len(pool[category])))
            if not pool[category]:
                del pool[category]
            _add_issue(issue)
            _remaining -= 1
            if pool:
                delay = TICKET_INTERVAL_SECONDS + random.uniform(-INTERVAL_JITTER_SECONDS, INTERVAL_JITTER_SECONDS)
                await asyncio.sleep(max(0.1, delay))
    finally:
        if generation == _generation:
            _stream_state = "finished"


@app.post("/mock/start")
def start_stream(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Porneste fluxul. Idempotent: daca ruleaza deja sau s-a terminat, nu face nimic."""
    global _stream_state
    if _stream_state != "idle":
        return {"started": False, "state": _stream_state}
    _stream_state = "running"
    background_tasks.add_task(_stream, _generation)
    return {"started": True, "state": "running"}


@app.post("/mock/reset")
def reset_store() -> dict[str, str]:
    global _generation, _stream_state, _remaining
    with _lock:
        _generation += 1
        _store.clear()
    _stream_state = "idle"
    _remaining = 0
    return {"status": "reset"}


@app.get("/mock/status")
def status() -> dict[str, Any]:
    with _lock:
        count = len(_store)
    return {"state": _stream_state, "tickets_in_store": count, "remaining": _remaining}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)