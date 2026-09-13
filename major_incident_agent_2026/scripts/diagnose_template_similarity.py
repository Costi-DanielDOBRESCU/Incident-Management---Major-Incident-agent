"""Diagnostic: pairwise BGE-M3 similarity between existing ticket templates.

For each root cause with 2+ hand-written templates, this builds the exact
composite text used by the detection pipeline (summary + description +
component + labels), embeds each template with BGE-M3, and prints the
pairwise cosine similarity matrix — so we can see exactly which template
pairs fall below the clustering threshold (0.75), before deciding how to
bridge them.

Run:
    python -m scripts.diagnose_template_similarity
"""

from __future__ import annotations

import sys
from pathlib import Path

from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.generate_mock_data import SERVICE_CAUSES  # noqa: E402

MODEL_NAME = "bge-m3"
SIMILARITY_THRESHOLD = 0.75
DUMMY_TIME = "10:00"  # placeholder for {time}, identical for all templates


def build_template_text(summary: str, description: str, service: str, category: str) -> str:
    """Mirror evaluate_detection.py's build_ticket_text(), but for a raw template."""
    filled_description = description.replace("{time}", DUMMY_TIME)

    return (
        f"Summary: {summary}\n"
        f"Description: {filled_description}\n"
        f"Component: {service}\n"
        f"Labels: {category.lower()}"
    )


def main() -> None:
    settings = get_settings()
    client = Client(host=settings.ollama_base_url)

    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print()

    weak_pairs_report: list[tuple[str, str, str, str, float]] = []

    for service, service_info in SERVICE_CAUSES.items():
        category = service_info["category"]

        for cause_key, cause_info in service_info["causes"].items():
            templates = cause_info["templates"]

            if len(templates) < 2:
                continue

            texts = [
                build_template_text(summary, description, service, category)
                for summary, description in templates
            ]

            embeddings = []
            for text in texts:
                response = client.embed(model=MODEL_NAME, input=text)
                embeddings.append(response["embeddings"][0])

            similarity_matrix = cosine_similarity(embeddings)

            print("=" * 70)
            print(f"Service: {service} | Cause: {cause_key}")
            print("=" * 70)

            labels = [summary for summary, _ in templates]

            for i, label in enumerate(labels):
                print(f"  [{i}] {label}")
            print()

            header = "      " + "".join(f"[{i}]".rjust(7) for i in range(len(labels)))
            print(header)

            for i in range(len(labels)):
                row = f"  [{i}] "
                for j in range(len(labels)):
                    value = similarity_matrix[i][j]
                    marker = "*" if (i != j and value < SIMILARITY_THRESHOLD) else " "
                    row += f"{value:.3f}{marker}".rjust(7)
                print(row)

                for j in range(i + 1, len(labels)):
                    if similarity_matrix[i][j] < SIMILARITY_THRESHOLD:
                        weak_pairs_report.append(
                            (service, cause_key, labels[i], labels[j], similarity_matrix[i][j])
                        )

            print()

    print("=" * 70)
    print("SUMMARY: pairs below threshold (need bridging)")
    print("=" * 70)

    if not weak_pairs_report:
        print("None — all template pairs are already above threshold.")
    else:
        weak_pairs_report.sort(key=lambda item: item[4])
        for service, cause_key, label_a, label_b, score in weak_pairs_report:
            print(f"{score:.3f}  [{service} / {cause_key}]  '{label_a}'  <->  '{label_b}'")


if __name__ == "__main__":
    main()