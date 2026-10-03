"""
scripts/eval_clusters.py

Evaluare la NIVEL DE CLUSTER (nu de tichet) pentru detectie, pe baza incident_group_id din ground_truth.json.
Raspunde la: fiecare incident real a fost detectat intact, spart in bucati, lipit de altul sau contaminat cu zgomot?

Foloseste aceleasi functii ca scripts/evaluate_detection.py (fereastra glisanta, clustere maximale)
si cache-ul de embeddings din scripts/sweep_detection.py (deci ruleaza rapid daca sweep-ul a rulat deja).

Compara, pentru fiecare prag, clusterele "brute" (cum le da fereastra glisanta de 20 min) cu cele
"cu merge" (clustere care impart tichete se unesc - acelasi incident vazut din ferestre diferite).

Rulare (din radacina proiectului, cu Ollama pornit si venv activ):
    python -m scripts.eval_clusters
    python -m scripts.eval_clusters --variant C --thresholds 0.70 0.75
"""

from __future__ import annotations

import argparse
from collections import Counter

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from app.detection.time_window import _get_created_at
from scripts.evaluate_detection import (
    MIN_CLUSTER_SIZE,
    WINDOW_MINUTES,
    keep_maximal_clusters,
    load_data,
    run_sliding_window_clustering,
)
from scripts.sweep_detection import VARIANTS, _key, _load_cache, ensure_embeddings


def merge_overlapping(clusters: list[frozenset[str]]) -> list[frozenset[str]]:
    """Uneste clusterele care au cel putin un tichet comun (acelasi incident vazut din ferestre diferite)."""
    merged: list[set[str]] = []
    for c in clusters:
        current = set(c)
        rest = []
        for m in merged:
            if m & current:
                current |= m
            else:
                rest.append(m)
        rest.append(current)
        merged = rest
    return [frozenset(m) for m in merged]


def evaluate(clusters: list[frozenset[str]], gt_by_ticket: dict[str, dict], incidents: dict[str, set[str]],
             cause_of: dict[str, str], verbose: bool) -> dict:
    # compozitia fiecarui cluster gasit
    comp = []
    for c in clusters:
        counts: Counter = Counter()
        for t in c:
            g = gt_by_ticket[t]
            if g["incident_group_id"]:
                counts[g["incident_group_id"]] += 1
            elif g["is_false_positive_trap"]:
                counts["TRAP"] += 1
            else:
                counts["NOISE"] += 1
        comp.append((c, counts))

    n_intact = n_fragmented = n_merged = n_missed = 0
    rows = []
    for gid, tickets in sorted(incidents.items()):
        size = len(tickets)
        touching = [(c, cnt) for c, cnt in comp if cnt.get(gid, 0) >= MIN_CLUSTER_SIZE]
        if not touching:
            n_missed += 1
            rows.append((gid, cause_of[gid], size, "MISSED", 0, 0.0, 0.0, 0, ""))
            continue
        main_c, main_cnt = max(touching, key=lambda x: x[1][gid])
        coverage = main_cnt[gid] / size
        purity = main_cnt[gid] / len(main_c)
        fragments = len(touching)
        merged_with = sorted(g for g, n in main_cnt.items() if g.startswith("GT-") and g != gid and n >= 2)
        if merged_with:
            status = "MERGED"; n_merged += 1
        elif fragments > 1 or coverage < 0.9:
            status = "FRAGMENTED"; n_fragmented += 1
        else:
            status = "INTACT"; n_intact += 1
        rows.append((gid, cause_of[gid], size, status, main_cnt[gid], coverage, purity, fragments, ",".join(merged_with)))

    # clustere care nu sunt un incident
    trap_clusters = [cnt for _, cnt in comp if cnt.most_common(1)[0][0] == "TRAP"]
    noise_clusters = [cnt for _, cnt in comp if cnt.most_common(1)[0][0] == "NOISE"]
    contaminated = [cnt for _, cnt in comp
                    if cnt.most_common(1)[0][0].startswith("GT-") and (cnt.get("NOISE", 0) + cnt.get("TRAP", 0)) > 0]

    if verbose:
        print(f"  {'Incident':<10}{'Cauza':<28}{'Tichete':>8}  {'Status':<11}{'In clusterul principal':>23}{'Acoperire':>11}{'Puritate':>10}{'Bucati':>8}  Lipit de")
        for gid, cause, size, status, inmain, cov, pur, frag, merged in rows:
            print(f"  {gid:<10}{cause:<28}{size:>8}  {status:<11}{inmain:>23}{cov:>11.0%}{pur:>10.0%}{frag:>8}  {merged}")

    return dict(intact=n_intact, fragmented=n_fragmented, merged=n_merged, missed=n_missed,
                total_clusters=len(clusters), trap_clusters=len(trap_clusters),
                noise_clusters=len(noise_clusters), contaminated=len(contaminated))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="C", help="prima litera a variantei din sweep_detection (A/B/C)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.65, 0.675, 0.70, 0.725, 0.75])
    ap.add_argument("--verbose-for", type=float, default=None, help="afiseaza tabelul detaliat doar pentru acest prag")
    args = ap.parse_args()

    name, build_text = next((n, f) for n, f in VARIANTS.items() if n.startswith(args.variant.upper()))
    client = Client(host=get_settings().ollama_base_url)
    ground_truth, tickets = load_data()
    tickets_sorted = sorted(tickets, key=_get_created_at)

    gt_by_ticket = {g["ticket_id"]: g for g in ground_truth}
    incidents: dict[str, set[str]] = {}
    cause_of: dict[str, str] = {}
    for g in ground_truth:
        if g["incident_group_id"]:
            incidents.setdefault(g["incident_group_id"], set()).add(g["ticket_id"])
            cause_of[g["incident_group_id"]] = g["root_cause_id"]
    n_trap_events = sum(1 for g in ground_truth if g["is_false_positive_trap"])

    cache = _load_cache()
    texts = [build_text(t) for t in tickets_sorted]
    ensure_embeddings(client, texts, cache)
    matrix = cosine_similarity([cache[_key(t)] for t in texts]).tolist()

    print(f"Varianta: {name}")
    print(f"Incidente reale: {len(incidents)} | tichete-capcana: {n_trap_events} | fereastra {WINDOW_MINUTES} min | min cluster {MIN_CLUSTER_SIZE}\n")

    summary = []
    for th in args.thresholds:
        found = run_sliding_window_clustering(tickets_sorted, matrix, WINDOW_MINUTES, th, MIN_CLUSTER_SIZE)
        raw = keep_maximal_clusters(found)
        merged = merge_overlapping(raw)
        show = args.verbose_for is not None and abs(th - args.verbose_for) < 1e-9
        for label, cl in (("brut", raw), ("cu merge", merged)):
            if show:
                print(f"--- Detalii prag {th:.3f} ({label}) ---")
            res = evaluate(cl, gt_by_ticket, incidents, cause_of, verbose=show)
            summary.append((th, label, res))
            if show:
                print()

    print(f"{'Prag':>6}  {'Mod':<9}{'Clustere':>9}{'Intacte':>9}{'Sparte':>8}{'Lipite':>8}{'Ratate':>8}{'Cl.-capcana':>13}{'Cl.-zgomot':>12}{'Incid+zgomot':>14}")
    for th, label, r in summary:
        print(f"{th:>6.3f}  {label:<9}{r['total_clusters']:>9}{r['intact']:>9}{r['fragmented']:>8}{r['merged']:>8}{r['missed']:>8}"
              f"{r['trap_clusters']:>13}{r['noise_clusters']:>12}{r['contaminated']:>14}")


if __name__ == "__main__":
    main()