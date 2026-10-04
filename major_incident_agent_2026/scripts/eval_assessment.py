"""
scripts/eval_assessment.py

Evalueaza Assessment Agent (LLM + RAG) pe TOATE clusterele din ground truth, in loc sa astepti
fluxuri aleatorii din simulator:
  - cele 18 incidente reale (GT-001..018)     -> asteptat: candidat major incident = True
  - capcanele cu >= 3 tichete (cauze individuale) -> asteptat: candidat major incident = False

Clusterele se construiesc din ground truth, cu fereastra reala a fiecarui grup (nu cea comprimata
a simulatorului), deci LLM-ul vede durate realiste (10-30 min). Detectia e evaluata separat de
scripts/eval_clusters.py; aici masuram doar judecata Assessment Agent.

Rezultatele detaliate se salveaza in output/eval_assessment.json.

Rulare (din radacina proiectului, cu Ollama pornit si venv activ):
    python -m scripts.eval_assessment
    python -m scripts.eval_assessment --only traps      # doar capcanele
    python -m scripts.eval_assessment --only incidents  # doar incidentele
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.agents.assessment_agent import AssessmentError, assess_incident
from app.config import get_settings
from app.detection.build_cluster import build_incident_cluster
from app.detection.text import build_ticket_text
from scripts.evaluate_detection import load_data
from scripts.sweep_detection import _key, _load_cache, ensure_embeddings

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "output" / "eval_assessment.json"
MIN_TRAP_SIZE = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["incidents", "traps"], default=None)
    args = ap.parse_args()

    client = Client(host=get_settings().ollama_base_url)
    ground_truth, tickets = load_data()
    by_key = {t["key"]: t for t in tickets}

    # grupuri de evaluat: (eticheta, tip, cauza, tichete)
    incidents: dict[str, list[str]] = defaultdict(list)
    traps: dict[str, list[str]] = defaultdict(list)
    cause_of: dict[str, str] = {}
    for g in ground_truth:
        if g["incident_group_id"]:
            incidents[g["incident_group_id"]].append(g["ticket_id"])
            cause_of[g["incident_group_id"]] = g["root_cause_id"]
        elif g["is_false_positive_trap"]:
            traps[g["root_cause_id"]].append(g["ticket_id"])

    cases: list[tuple[str, str, str, list[str]]] = []
    if args.only != "traps":
        cases += [(gid, "incident", cause_of[gid], keys) for gid, keys in sorted(incidents.items())]
    if args.only != "incidents":
        cases += [(f"TRAP:{cause}", "trap", cause, keys) for cause, keys in sorted(traps.items()) if len(keys) >= MIN_TRAP_SIZE]

    # embeddings (din cache) pentru similaritatea medie din cluster
    cache = _load_cache()
    all_keys = sorted({k for _, _, _, keys in cases for k in keys})
    texts = {k: build_ticket_text(by_key[k]) for k in all_keys}
    ensure_embeddings(client, list(texts.values()), cache)

    results = []
    print(f"Se evalueaza {len(cases)} clustere...\n")
    print(f"{'Caz':<34}{'Tip':<10}{'Asteptat':<10}{'Sev':<6}{'Candidat':<10}{'Actiune':<24}{'Conf':>5}  OK?")

    for seq, (label, kind, cause, keys) in enumerate(cases, start=1):
        group = sorted((by_key[k] for k in keys), key=lambda t: t["fields"]["created"])
        matrix = cosine_similarity([cache[_key(build_ticket_text(t))] for t in group]).tolist()
        cluster = build_incident_cluster(group, matrix, list(range(len(group))), seq)
        summaries = [t["fields"]["summary"] for t in group]

        expected = kind == "incident"
        started = time.time()
        try:
            a = assess_incident(cluster, summaries)
        except AssessmentError as exc:
            print(f"{label:<34}{kind:<10}{str(expected):<10}ERROR: {exc}")
            results.append(dict(case=label, kind=kind, cause=cause, expected_candidate=expected, error=str(exc)))
            continue

        ok = a.is_major_incident_candidate == expected
        print(f"{label:<34}{kind:<10}{str(expected):<10}{a.estimated_severity:<6}{str(a.is_major_incident_candidate):<10}"
              f"{a.recommended_action:<24}{a.confidence:>5.2f}  {'OK' if ok else 'GRESIT'}  ({time.time() - started:.0f}s)")
        results.append(dict(
            case=label, kind=kind, cause=cause, service=cluster.service_guess, ticket_count=cluster.ticket_count,
            window_minutes=round((cluster.window_end - cluster.window_start).total_seconds() / 60, 1),
            expected_candidate=expected, severity=a.estimated_severity,
            is_candidate=a.is_major_incident_candidate, action=a.recommended_action, confidence=a.confidence,
            correct=ok, rag_sources=a.rag_sources, reasoning=a.reasoning,
        ))

    scored = [r for r in results if "correct" in r]
    inc = [r for r in scored if r["kind"] == "incident"]
    trp = [r for r in scored if r["kind"] == "trap"]
    print("\nRezumat")
    print("-------")
    if inc:
        print(f"Incidente propuse corect:  {sum(r['correct'] for r in inc)}/{len(inc)}")
    if trp:
        print(f"Capcane respinse corect:   {sum(r['correct'] for r in trp)}/{len(trp)}")
    wrong = [r["case"] for r in scored if not r["correct"]]
    print(f"Cazuri gresite: {wrong if wrong else 'niciunul'}")
    errors = [r["case"] for r in results if "error" in r]
    if errors:
        print(f"Erori LLM: {errors}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetalii salvate in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()