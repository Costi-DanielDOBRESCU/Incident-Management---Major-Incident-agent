"""
scripts/sweep_detection.py

Compara variante de text pentru embeddings si praguri de similaritate, folosind EXACT
evaluarea din scripts/evaluate_detection.py (fereastra glisanta de 20 min, clustere maximale,
Precision/Recall/F1 pe ground_truth.json, plus capcanele de fals-pozitiv).

Embeddings-urile se pastreaza in cache pe disc (app/data/embed_cache.pkl), deci a doua
rulare dureaza secunde, nu minute.

Rulare (din radacina proiectului, cu Ollama pornit si venv activ):
    python -m scripts.sweep_detection
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from app.detection.time_window import _get_created_at
from scripts.evaluate_detection import (
    MIN_CLUSTER_SIZE,
    MODEL_NAME,
    WINDOW_MINUTES,
    build_ticket_text,
    calculate_metrics,
    keep_maximal_clusters,
    load_data,
    run_sliding_window_clustering,
)

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
BATCH_SIZE = 32
CACHE_FILE = Path(__file__).resolve().parent.parent / "app" / "data" / "embed_cache.pkl"


# ---------------------------------------------------------------------------
# Variante de text trimis la embeddings
# ---------------------------------------------------------------------------
def _component(ticket: dict) -> str:
    comps = ticket["fields"].get("components") or []
    return comps[0]["name"] if comps else "Unknown"


def text_summary_only(ticket: dict) -> str:
    return str(ticket["fields"].get("summary", ""))


def text_service_summary(ticket: dict) -> str:
    return f"Component: {_component(ticket)}\nSummary: {ticket['fields'].get('summary', '')}"


VARIANTS = {
    "A. doar summary (folosit acum in UI)": text_summary_only,
    "B. serviciu + summary": text_service_summary,
    "C. summary+description+component+labels (cel evaluat in Etapa 3)": build_ticket_text,
}


# ---------------------------------------------------------------------------
# Cache embeddings
# ---------------------------------------------------------------------------
def _key(text: str) -> str:
    return hashlib.sha1(f"{MODEL_NAME}\n{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as f:
        pickle.dump(cache, f)


def ensure_embeddings(client: Client, texts: list[str], cache: dict[str, list[float]]) -> None:
    missing = list({t for t in texts if _key(t) not in cache})
    if not missing:
        return
    print(f"  se calculeaza {len(missing)} embeddings noi (restul sunt in cache)...")
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        response = client.embed(model=MODEL_NAME, input=batch)
        for text, emb in zip(batch, response["embeddings"]):
            cache[_key(text)] = emb
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def main() -> None:
    client = Client(host=get_settings().ollama_base_url)
    ground_truth, tickets = load_data()
    tickets_sorted = sorted(tickets, key=_get_created_at)

    traps = {g["ticket_id"] for g in ground_truth if g["is_false_positive_trap"]}
    n_major = sum(g["is_major_incident_ticket"] for g in ground_truth)

    print(f"Tichete: {len(tickets_sorted)} | tichete de incident major: {n_major} | capcane fals-pozitiv: {len(traps)}")
    print(f"Fereastra: {WINDOW_MINUTES} min | min cluster: {MIN_CLUSTER_SIZE} | model: {MODEL_NAME}")

    cache = _load_cache()
    rows: list[dict] = []

    for variant_name, build_text in VARIANTS.items():
        print(f"\n{variant_name}")
        texts = [build_text(t) for t in tickets_sorted]
        ensure_embeddings(client, texts, cache)
        embeddings = [cache[_key(t)] for t in texts]
        matrix = cosine_similarity(embeddings).tolist()

        for th in THRESHOLDS:
            found = run_sliding_window_clustering(
                tickets_sorted=tickets_sorted,
                similarity_matrix=matrix,
                window_minutes=WINDOW_MINUTES,
                threshold=th,
                min_cluster_size=MIN_CLUSTER_SIZE,
            )
            clusters = keep_maximal_clusters(found)
            predicted: set[str] = set()
            for c in clusters:
                predicted.update(c)

            tp, fp, fn, precision, recall, f1 = calculate_metrics(ground_truth, predicted)
            rows.append({
                "variant": variant_name[0], "name": variant_name, "th": th,
                "clusters": len(clusters), "tp": tp, "fp": fp, "fn": fn,
                "p": precision, "r": recall, "f1": f1, "traps": len(predicted & traps),
            })

    print("\n" + "=" * 86)
    print(f"{'Var':<4}{'Prag':>6}{'Clustere':>10}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Capcane':>9}")
    print("=" * 86)
    last = None
    for r in rows:
        if last and r["variant"] != last:
            print("-" * 86)
        last = r["variant"]
        print(
            f"{r['variant']:<4}{r['th']:>6.2f}{r['clusters']:>10}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
            f"{r['p']:>8.3f}{r['r']:>8.3f}{r['f1']:>8.3f}{r['traps']:>9}"
        )

    best = sorted(rows, key=lambda r: (-r["f1"], r["traps"], -r["th"]))[:3]
    print("\nTop 3 (dupa F1, apoi capcane detectate, apoi prag mai mare):")
    for r in best:
        print(f"  {r['variant']} | prag {r['th']:.2f} | F1 {r['f1']:.3f} | P {r['p']:.3f} | R {r['r']:.3f} | capcane {r['traps']}")


if __name__ == "__main__":
    main()