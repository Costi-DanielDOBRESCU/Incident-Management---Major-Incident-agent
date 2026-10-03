"""
app/ingestion/mock_jira_live.py

Mock Jira LIVE: tichetele "sosesc" in timp real, nu sunt servite dintr-un fisier static.

- Store in memorie cu tichetele primite. `created` = momentul primirii (ca in Jira real).
- GET  /rest/api/2/search?since=&limit=  -> acelasi format ca mock-ul static ({"total", "issues"}).
- POST /rest/api/2/issue                 -> creeaza un tichet (ca in Jira real).
- POST /mock/start                      -> porneste fluxul automat de tichete (idempotent).
- POST /mock/reset                      -> opreste fluxul si goleste store-ul.
- GET  /mock/status, /health

Simulatorul emite TICKETS_PER_RUN tichete la intervale de ~TICKET_INTERVAL_SECONDS, folosind
datele din tickets.json + ground_truth.json. Un flux contine:
  * o categorie principala aleasa aleatoriu: un incident real (GT-xxx) sau, cu probabilitatea
    TRAP_PROBABILITY, o capcana de fals-pozitiv (cauza individuala cu formulari asemanatoare);
    din ea se emit PRIMARY_TICKETS tichete aleatorii (sau toate, daca sunt mai putine);
  * restul, zgomot de fundal: tichete izolate din ALTE servicii, cate unul per serviciu, ca sa
    nu formeze clustere accidentale.
Ordinea emiterii e aleatorie. Niciun tichet nu e emis de doua ori pe parcursul unui flux.

mock_jira_api.py (static) ramane neschimbat - il folosesc testele existente.

Rulare (din radacina proiectului, cu venv activ):
    uvicorn app.ingestion.mock_jira_live:app --port 8002
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.ingestion.mock_jira_api import (
    TICKETS_FILE,
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
TICKETS_PER_RUN = 8              # cate tichete se trimit intr-un flux
PRIMARY_TICKETS = 6              # din care: tichete din categoria principala (incident/capcana)
TRAP_PROBABILITY = 0.35          # sansa ca un flux sa aiba ca principala o capcana, nu un incident
GROUND_TRUTH_FILE = TICKETS_FILE.parent / "ground_truth.json"

# ---------------------------------------------------------------------------
# Store in memorie
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_store: list[dict[str, Any]] = []
_generation = 0          # creste la /mock/stop si /mock/reset -> replay-urile vechi se opresc
_stream_state = "idle"  # idle | running | finished
_remaining = 0
_current_run: dict[str, Any] = {}   # adevarul (ground truth) pentru fluxul curent, doar pentru verificari
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
def _build_run() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compune un flux de TICKETS_PER_RUN tichete: categoria principala (incident sau capcana)
    + zgomot din alte servicii. Intoarce (tichete in ordinea emiterii, descrierea fluxului).
    """
    issues = {i["key"]: i for i in _load_issues()}
    truth = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))

    def service(issue: dict[str, Any]) -> str:
        comps = issue["fields"].get("components") or []
        return comps[0]["name"] if comps else "Unknown"

    incidents: dict[str, list[dict[str, Any]]] = {}
    traps: dict[str, list[dict[str, Any]]] = {}
    noise: list[dict[str, Any]] = []
    for g in truth:
        issue = issues.get(g["ticket_id"])
        if issue is None:
            continue
        if g["incident_group_id"]:
            incidents.setdefault(g["incident_group_id"], []).append(issue)
        elif g["is_false_positive_trap"]:
            traps.setdefault(g["root_cause_id"], []).append(issue)
        else:
            noise.append(issue)

    min_size = 3  # sub acest numar un grup nu poate forma cluster
    trap_options = {k: v for k, v in traps.items() if len(v) >= min_size}
    use_trap = bool(trap_options) and random.random() < TRAP_PROBABILITY
    kind, pool = ("trap", trap_options) if use_trap else ("incident", incidents)

    category = random.choice(list(pool))
    members = pool[category]
    primary = random.sample(members, k=min(PRIMARY_TICKETS, len(members)))
    primary_service = service(primary[0])

    # zgomot: tichete din alte servicii, cate unul per serviciu
    n_noise = max(0, TICKETS_PER_RUN - len(primary))
    by_service: dict[str, list[dict[str, Any]]] = {}
    for issue in noise:
        if service(issue) != primary_service:
            by_service.setdefault(service(issue), []).append(issue)
    chosen_services = random.sample(list(by_service), k=min(n_noise, len(by_service)))
    filler = [random.choice(by_service[s_]) for s_ in chosen_services]

    emitted = primary + filler
    random.shuffle(emitted)

    info = {
        "kind": kind,
        "group": category,
        "service": primary_service,
        "root_cause": truth_cause(truth, primary[0]["key"]),
        "primary_tickets": sorted(i["key"] for i in primary),
        "noise_tickets": sorted(i["key"] for i in filler),
    }
    return emitted, info


def truth_cause(truth: list[dict[str, Any]], key: str) -> str:
    return next((g["root_cause_id"] for g in truth if g["ticket_id"] == key), "")


async def _stream(generation: int) -> None:
    global _stream_state, _remaining, _current_run
    run, info = _build_run()
    _current_run = info
    _remaining = len(run)
    try:
        for position, issue in enumerate(run):
            if generation != _generation:
                return
            _add_issue(issue)
            _remaining -= 1
            if position < len(run) - 1:
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
    global _generation, _stream_state, _remaining, _current_run
    with _lock:
        _generation += 1
        _store.clear()
    _stream_state = "idle"
    _remaining = 0
    _current_run = {}
    return {"status": "reset"}


@app.get("/mock/status")
def status() -> dict[str, Any]:
    with _lock:
        count = len(_store)
    return {"state": _stream_state, "tickets_in_store": count, "remaining": _remaining, "run": _current_run}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)