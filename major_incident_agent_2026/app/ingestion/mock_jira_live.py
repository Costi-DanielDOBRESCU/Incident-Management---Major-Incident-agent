"""
app/ingestion/mock_jira_live.py

Mock Jira LIVE: tichetele "sosesc" in timp real, nu sunt servite dintr-un fisier static.

- Store in memorie cu tichetele primite. `created` = momentul primirii (ca in Jira real).
- GET  /rest/api/2/search?since=&limit=  -> acelasi format ca mock-ul static ({"total", "issues"}).
- POST /rest/api/2/issue                 -> creeaza un tichet (ca in Jira real).
- POST /mock/replay                      -> simulator: reda un scenariu tichet cu tichet.
- POST /mock/stop                        -> opreste replay-urile in curs.
- POST /mock/reset                       -> opreste replay-urile si goleste store-ul.
- GET  /mock/scenarios, /mock/status, /health

mock_jira_api.py (static) ramane neschimbat - il folosesc testele existente.

Rulare (din radacina proiectului, cu venv activ):
    uvicorn app.ingestion.mock_jira_live:app --port 8002
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

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
# Store in memorie
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_store: list[dict[str, Any]] = []
_generation = 0          # creste la /mock/stop si /mock/reset -> replay-urile vechi se opresc
_active_replays = 0
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
class ReplayRequest(BaseModel):
    scenario: Optional[str] = Field(default=None, description="Cheie din /mock/scenarios")
    ticket_keys: Optional[list[str]] = Field(default=None, description="Alternativ: lista explicita de chei")
    interval_seconds: float = Field(default=1.5, ge=0.0, le=60.0)


async def _replay(issues: list[dict[str, Any]], interval: float, generation: int) -> None:
    global _active_replays
    _active_replays += 1
    try:
        for issue in issues:
            if generation != _generation:
                return
            _add_issue(issue)
            await asyncio.sleep(interval)
    finally:
        _active_replays -= 1


@app.get("/mock/scenarios")
def list_scenarios() -> dict[str, int]:
    return {name: len(keys) for name, keys in DEMO_MAJOR_INCIDENTS.items()}


@app.post("/mock/replay")
def start_replay(
    background_tasks: BackgroundTasks,
    req: ReplayRequest = Body(...),
) -> dict[str, Any]:
    if req.ticket_keys:
        keys = req.ticket_keys
    elif req.scenario:
        if req.scenario not in DEMO_MAJOR_INCIDENTS:
            raise HTTPException(status_code=404, detail=f"Scenariu necunoscut: {req.scenario}")
        keys = DEMO_MAJOR_INCIDENTS[req.scenario]
    else:
        raise HTTPException(status_code=422, detail="Da 'scenario' sau 'ticket_keys'.")

    by_key = {i["key"]: i for i in _load_issues()}
    issues = [by_key[k] for k in keys if k in by_key]
    if not issues:
        raise HTTPException(status_code=404, detail="Niciun tichet gasit pentru cheile date.")
    issues.sort(key=lambda i: _parse_jira_datetime(i["fields"]["created"]))

    background_tasks.add_task(_replay, issues, req.interval_seconds, _generation)
    return {"started": True, "tickets": len(issues), "interval_seconds": req.interval_seconds}


@app.post("/mock/stop")
def stop_replays() -> dict[str, str]:
    global _generation
    _generation += 1
    return {"status": "stopped"}


@app.post("/mock/reset")
def reset_store() -> dict[str, str]:
    global _generation
    with _lock:
        _generation += 1
        _store.clear()
    return {"status": "reset"}


@app.get("/mock/status")
def status() -> dict[str, Any]:
    with _lock:
        count = len(_store)
    return {"tickets_in_store": count, "active_replays": _active_replays}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)