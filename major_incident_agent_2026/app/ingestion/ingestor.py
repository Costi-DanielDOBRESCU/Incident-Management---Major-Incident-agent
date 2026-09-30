"""
app/ingestion/ingestor.py

Ingestor: interogheaza periodic API-ul Jira-like (mock live sau Jira real) si
salveaza tichetele noi in SQLite (app/data/tickets.db).

Ruleaza ca proces SEPARAT, in afara grafului LangGraph.

Rulare (din radacina proiectului, cu venv activ):
    python -m app.ingestion.ingestor                 # poll la 3s
    python -m app.ingestion.ingestor --interval 5
    python -m app.ingestion.ingestor --reset         # goleste tabela locala si porneste de la zero
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import httpx

from app.config import get_settings
from app.ingestion import ticket_store

PAGE_SIZE = 500  # maximul acceptat de API


def poll_once(client: httpx.Client, base_url: str) -> int:
    """
    Un ciclu de polling: cere tichetele create de la watermark incoace si le salveaza.
    Intoarce numarul de tichete NOI. Repeta cat timp pagina vine plina.
    """
    total_new = 0
    while True:
        params: dict[str, object] = {"limit": PAGE_SIZE}
        watermark: Optional[str] = ticket_store.get_watermark()
        if watermark:
            params["since"] = watermark

        resp = client.get(f"{base_url}/rest/api/2/search", params=params)
        resp.raise_for_status()
        issues = resp.json()["issues"]

        new_issues = ticket_store.upsert_tickets(issues)
        for issue in new_issues:
            f = issue["fields"]
            comps = f.get("components") or [{"name": "Unknown"}]
            print(f"  + {issue['key']}  [{comps[0]['name']}]  {f.get('summary', '')}")
        total_new += len(new_issues)

        # Pagina plina si cu tichete noi -> mai poate fi continuare
        if len(issues) < PAGE_SIZE or not new_issues:
            break
    return total_new


def run(interval: float, base_url: str) -> None:
    ticket_store.init_db()
    print(f"Ingestor pornit | sursa: {base_url} | interval: {interval}s | db: {ticket_store.DB_PATH}")
    print(f"Tichete deja in DB: {ticket_store.count_tickets()} | watermark: {ticket_store.get_watermark()}")

    with httpx.Client(timeout=10) as client:
        while True:
            try:
                n = poll_once(client, base_url)
                if n:
                    print(f"[{time.strftime('%H:%M:%S')}] {n} tichete noi | total in DB: {ticket_store.count_tickets()}")
            except httpx.HTTPError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] Sursa indisponibila ({exc.__class__.__name__}); reincerc...")
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestor tichete Jira -> SQLite")
    parser.add_argument("--interval", type=float, default=3.0, help="secunde intre polling-uri")
    parser.add_argument("--url", default=None, help="URL de baza al API-ului (implicit: mock live din config)")
    parser.add_argument("--reset", action="store_true", help="goleste tabela locala inainte de pornire")
    args = parser.parse_args()

    base_url = args.url or f"http://127.0.0.1:{get_settings().mock_jira_port}"

    if args.reset:
        ticket_store.reset_db()
        print("Baza locala de tichete a fost golita.")

    try:
        run(args.interval, base_url)
    except KeyboardInterrupt:
        print("\nIngestor oprit.")


if __name__ == "__main__":
    main()