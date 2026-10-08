"""
Evaluare de RETRIEVAL pentru Assessment Agent (fara LLM, gratuita, rapida).

Intrebare: documentele care conteaza pentru decizie ajung efectiv in prompt?
Assessment ia doar top-2 runbook-uri si top-3 post-mortemuri, filtrate pe serviciu, iar serviciile au 3-4
runbook-uri. Daca runbook-ul cu criteriile de severitate sau cel cu "known non-incident patterns" nu intra
in top-2, LLM-ul decide fara el.

Pentru fiecare dintre cele 26 de clustere din ground truth (18 incidente + 8 capcane), scriptul reproduce
EXACT interogarea din Assessment (acelasi text, acelasi filtru de serviciu) si raporteaza:
  - rangul runbook-ului cu criterii de severitate (heuristica: mentioneaza cel putin doua niveluri SEV);
  - rangul runbook-ului cu tipar non-incident (heuristica: "do not declare" / "non-incident" / "request wave, not an incident",
    dar nu runbook-ul de severitate, care poate mentiona "not an incident" in treacat);
  - recall@k pentru k = 1, 2, 3 si toate runbook-urile serviciului (ce-ar fi daca am creste n_results).

Heuristicile sunt in `is_severity_doc` / `is_non_incident_doc`; ajusteaza-le daca formatul runbook-urilor difera.

Rulare:
    python -m scripts.eval_retrieval
    python -m scripts.eval_retrieval --only traps
    python -m scripts.eval_retrieval --runbook-k 3
"""

from __future__ import annotations

import re
from typing import Any, Callable

COLLECTION_HISTORICAL = "historical_major_incidents"
COLLECTION_RUNBOOKS = "runbooks"
MIN_TRAP_SIZE = 3
KS = (1, 2, 3)  # pentru recall@k; se adauga mereu si "toate"

QueryFn = Callable[[str, str, int, str], list[dict[str, Any]]]


def is_severity_doc(doc: dict[str, Any]) -> bool:
    """Runbook cu criterii de severitate: mentioneaza cel putin doua niveluri SEV diferite."""
    return len(set(re.findall(r"SEV[123]", doc.get("content", "")))) >= 2


_NON_INCIDENT_MARKERS = ("non-incident", "do not declare", "request wave, not an incident")


def is_non_incident_doc(doc: dict[str, Any]) -> bool:
    """
    Runbook care descrie un tipar non-incident (ex. RB-VPN-009: "request wave, not an incident: do not declare").
    Runbook-ul de severitate e exclus: poate mentiona in treacat "not an incident" (ex. RB-VPN-002).
    """
    text = doc.get("content", "").lower()
    return any(marker in text for marker in _NON_INCIDENT_MARKERS) and not is_severity_doc(doc)


def _best_rank(docs: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int | None:
    """Rangul (de la 1) al primului document care satisface predicatul; None daca nu exista."""
    for position, doc in enumerate(docs, start=1):
        if predicate(doc):
            return position
    return None


def analyse_case(
    query_text: str,
    service: str,
    query_fn: QueryFn,
    runbook_k: int = 2,
    postmortem_k: int = 3,
    max_docs: int = 50,
) -> dict[str, Any]:
    """
    Analizeaza retrieval-ul pentru un cluster. `query_fn(query, collection, n_results, service)`
    intoarce documentele ordonate dupa scor (ca query_knowledge_base).
    """
    runbooks = query_fn(query_text, COLLECTION_RUNBOOKS, max_docs, service)
    postmortems = query_fn(query_text, COLLECTION_HISTORICAL, max_docs, service)

    severity_rank = _best_rank(runbooks, is_severity_doc)
    non_incident_rank = _best_rank(runbooks, is_non_incident_doc)

    return {
        "service": service,
        "runbooks_total": len(runbooks),
        "runbooks_ranked": [(d["doc_id"], d["score"]) for d in runbooks],
        "postmortems_ranked": [(d["doc_id"], d["score"]) for d in postmortems],
        "severity_docs": [d["doc_id"] for d in runbooks if is_severity_doc(d)],
        "non_incident_docs": [d["doc_id"] for d in runbooks if is_non_incident_doc(d)],
        "retrieved_runbooks": [d["doc_id"] for d in runbooks[:runbook_k]],
        "retrieved_postmortems": [d["doc_id"] for d in postmortems[:postmortem_k]],
        "severity_rank": severity_rank,
        "non_incident_rank": non_incident_rank,
        "severity_in_top_k": severity_rank is not None and severity_rank <= runbook_k,
        "non_incident_in_top_k": non_incident_rank is not None and non_incident_rank <= runbook_k,
    }


def recall_at_k(ranks: list[int | None], k: int | None) -> float | None:
    """Fractiunea de cazuri cu rang <= k (k=None: orice rang, adica exista in serviciu). None daca nu sunt cazuri."""
    if not ranks:
        return None
    hits = sum(1 for r in ranks if r is not None and (k is None or r <= k))
    return hits / len(ranks)


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Recall@k pentru runbook-ul cu severitate si cel cu non-incident, pe lista de rezultate."""
    summary: dict[str, dict[str, float | None]] = {}
    for name, key in (("severity", "severity_rank"), ("non_incident", "non_incident_rank")):
        ranks = [r[key] for r in results]
        summary[name] = {f"@{k}": recall_at_k(ranks, k) for k in KS}
        summary[name]["exista"] = recall_at_k(ranks, None)
    return summary


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def main() -> None:
    import argparse
    import json
    from collections import defaultdict
    from pathlib import Path

    from app.agents.assessment_agent import _build_rag_query_text
    from app.detection.build_cluster import build_incident_cluster
    from app.rag.query import query_knowledge_base
    from scripts.evaluate_detection import load_data

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["incidents", "traps"], default=None)
    ap.add_argument("--runbook-k", type=int, default=2, help="cate runbook-uri ia Assessment (implicit 2)")
    ap.add_argument("--postmortem-k", type=int, default=3, help="cate post-mortemuri ia Assessment (implicit 3)")
    args = ap.parse_args()

    ground_truth, tickets = load_data()
    by_key = {t["key"]: t for t in tickets}

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
        cases += [
            (f"TRAP:{cause}", "trap", cause, keys)
            for cause, keys in sorted(traps.items())
            if len(keys) >= MIN_TRAP_SIZE
        ]

    def query_fn(query: str, collection: str, n: int, service: str) -> list[dict[str, Any]]:
        from app.rag.chroma_client import get_or_create_collection

        available = len(get_or_create_collection(collection).get(where={"service": service})["ids"])
        if available == 0:
            return []
        return query_knowledge_base(query, collection, n_results=min(n, available), service_filter=service)

    print(f"Se evalueaza retrieval pentru {len(cases)} clustere (runbook top-{args.runbook_k}, post-mortem top-{args.postmortem_k})\n")
    print(f"{'Caz':<34}{'Tip':<10}{'Serviciu':<18}{'RB':>3} {'Sev#':>5} {'NonInc#':>8}  Runbook-uri luate")

    results: list[dict[str, Any]] = []
    for seq, (label, kind, cause, keys) in enumerate(cases, start=1):
        group = sorted((by_key[k] for k in keys), key=lambda t: t["fields"]["created"])
        n = len(group)
        # service_guess nu depinde de matrice; o matrice neutra e suficienta pentru a construi clusterul
        cluster = build_incident_cluster(group, [[1.0] * n for _ in range(n)], list(range(n)), seq)
        summaries = [t["fields"]["summary"] for t in group]
        query_text = _build_rag_query_text(cluster, summaries)

        analysis = analyse_case(
            query_text, cluster.service_guess, query_fn,
            runbook_k=args.runbook_k, postmortem_k=args.postmortem_k,
        )
        analysis.update(case=label, kind=kind, cause=cause)
        results.append(analysis)

        sev = analysis["severity_rank"] or "-"
        non = analysis["non_incident_rank"] or "-"
        flag = ""
        if not analysis["severity_in_top_k"]:
            flag += "  <- criterii severitate IN AFARA top-k"
        if kind == "trap" and not analysis["non_incident_in_top_k"]:
            flag += "  <- non-incident IN AFARA top-k"
        print(
            f"{label:<34}{kind:<10}{cluster.service_guess:<18}{analysis['runbooks_total']:>3} "
            f"{str(sev):>5} {str(non):>8}  {', '.join(analysis['retrieved_runbooks'])}{flag}"
        )

    print("\n=== Recall (fractiunea de cazuri in care documentul e in top-k) ===")
    for title, subset in (
        ("Incidente", [r for r in results if r["kind"] == "incident"]),
        ("Capcane", [r for r in results if r["kind"] == "trap"]),
    ):
        if not subset:
            continue
        s = summarize(subset)
        print(f"\n{title} ({len(subset)} cazuri)")
        for name, label in (("severity", "runbook criterii severitate"), ("non_incident", "runbook non-incident")):
            row = s[name]
            print(
                f"  {label:<30} @1 {_pct(row['@1']):>5}   @2 {_pct(row['@2']):>5}   @3 {_pct(row['@3']):>5}"
                f"   exista in serviciu {_pct(row['exista']):>5}"
            )

    print("\n=== Clasificarea runbook-urilor pe serviciu (verifica din ochi) ===")
    seen: set[str] = set()
    for r in results:
        if r["service"] in seen:
            continue
        seen.add(r["service"])
        all_ids = [doc_id for doc_id, _ in r["runbooks_ranked"]]
        other = [i for i in all_ids if i not in r["severity_docs"] and i not in r["non_incident_docs"]]
        print(
            f"  {r['service']:<18} severitate: {', '.join(r['severity_docs']) or '-':<14} "
            f"non-incident: {', '.join(r['non_incident_docs']) or '-':<14} altele: {', '.join(other) or '-'}"
        )

    out = Path(__file__).resolve().parent.parent / "output" / "eval_retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRezultate detaliate: {out}")


if __name__ == "__main__":
    main()