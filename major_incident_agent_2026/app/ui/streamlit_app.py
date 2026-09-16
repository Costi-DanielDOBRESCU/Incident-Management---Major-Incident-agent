"""
UI minimal (Streamlit) pentru demonstrarea fluxului MIA end-to-end.

v1: simulare temporala peste tichetele mock reale + Detection Pipeline live
(embeddings + clustering), afisat ca lista de tichete in fereastra curenta
si clustere candidate gasite.

v2: buton "Evalueaza" per cluster -> Assessment Agent (Etapa 5), afiseaza
severitate, decizie, reasoning + surse RAG.

Simularea temporala: un slider peste intervalul real de timestamp-uri din
tickets.json (2026-08-01 -> 2026-08-09). La fiecare pozitie, aplicatia
recalculeaza exact ce ar vedea sistemul in productie la acel moment:
tichetele din fereastra glisanta (settings.clustering_window_minutes),
urmate de embeddings + similaritate + clustering (Etapa 3, neschimbate).

Urmeaza (v3): Approve/Reject uman -> MajorIncident (human-in-the-loop, doc.
sectiunea 10).

Rulare: `python -m streamlit run app/ui/streamlit_app.py`
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from app.agents.assessment_agent import AssessmentError, assess_incident
from app.config import get_settings
from app.detection.build_cluster import build_incident_cluster
from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.detection.time_window import filter_tickets_by_time_window
from app.ingestion.mock_jira_api import fetch_tickets

st.set_page_config(page_title="MIA — Major Incident Agent", layout="wide")

settings = get_settings()


@st.cache_resource(show_spinner="Se incarca tichetele mock...")
def load_all_tickets() -> list[dict]:
    """Incarca o singura data toate tichetele mock (nu se schimba intre rulari)."""
    return fetch_tickets(limit=500)["issues"]


@st.cache_data(show_spinner=False)
def cached_embedding(text: str) -> list[float]:
    """
    Wrapper cache peste create_embedding() (Ollama). Cheia de cache e chiar
    textul - daca acelasi tichet apare in ferestre diferite (slider miscat),
    nu se recalculeaza embedding-ul de fiecare data.
    """
    return create_embedding(text)


def _ticket_created_at(ticket: dict) -> datetime:
    value = ticket["fields"]["created"]
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if value.endswith("+0000"):
        value = value[:-5] + "+00:00"
    return datetime.fromisoformat(value)


def _ticket_service(ticket: dict) -> str:
    components = ticket["fields"].get("components") or []
    return components[0]["name"] if components else "Unknown"


all_tickets = load_all_tickets()
all_created = [_ticket_created_at(t) for t in all_tickets]
min_dt_naive = min(all_created).replace(tzinfo=None)
max_dt_naive = max(all_created).replace(tzinfo=None)

st.title("🔎 MIA — Major Incident Agent (demo)")
st.caption(
    f"Simulare peste {len(all_tickets)} tichete mock "
    f"({min_dt_naive:%Y-%m-%d %H:%M} → {max_dt_naive:%Y-%m-%d %H:%M} UTC). "
    f"Fereastra de corelare: {settings.clustering_window_minutes} min."
)

selected_naive = st.slider(
    "⏱️ Timp simulat (momentul curent al sistemului)",
    min_value=min_dt_naive,
    max_value=max_dt_naive,
    value=min_dt_naive + timedelta(hours=14, minutes=21),  # ~ burst GT-001 (Internal Portal)
    step=timedelta(minutes=5),
    format="DD/MM HH:mm",
)
reference_time = selected_naive.replace(tzinfo=timezone.utc)

window_tickets = filter_tickets_by_time_window(
    tickets=all_tickets,
    reference_time=reference_time,
    window_minutes=settings.clustering_window_minutes,
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"📋 Tichete in fereastra curenta ({len(window_tickets)})")
    if not window_tickets:
        st.info("Niciun tichet in aceasta fereastra de timp.")
    else:
        table_rows = [
            {
                "ID": t["key"],
                "Ora": _ticket_created_at(t).strftime("%H:%M:%S"),
                "Serviciu": _ticket_service(t),
                "Rezumat": t["fields"]["summary"],
            }
            for t in sorted(window_tickets, key=_ticket_created_at)
        ]
        st.dataframe(table_rows, hide_index=True, use_container_width=True)

with col_right:
    st.subheader("🧩 Clustere candidate detectate")

    if len(window_tickets) < settings.min_tickets_per_cluster:
        st.info(f"Sub minimul de {settings.min_tickets_per_cluster} tichete — nu se ruleaza clustering.")
    else:
        with st.spinner("Se calculeaza embeddings + similaritate..."):
            texts = [t["fields"]["summary"] for t in window_tickets]
            embeddings = [cached_embedding(text) for text in texts]
            similarity_matrix = calculate_similarity_matrix(embeddings)
            raw_clusters = cluster_similar_tickets(
                similarity_matrix,
                threshold=settings.similarity_threshold,
                min_cluster_size=settings.min_tickets_per_cluster,
            )

        if not raw_clusters:
            st.info("Niciun cluster peste prag in aceasta fereastra.")
        else:
            for seq, indices in enumerate(raw_clusters, start=1):
                cluster = build_incident_cluster(window_tickets, similarity_matrix, indices, seq)
                cluster_tickets = [window_tickets[i] for i in indices]
                summaries = [t["fields"]["summary"] for t in cluster_tickets]

                with st.container(border=True):
                    st.markdown(f"**{cluster.cluster_id}** — {cluster.service_guess}")
                    st.caption(
                        f"{cluster.ticket_count} tichete · "
                        f"similaritate medie {cluster.centroid_similarity} · "
                        f"{cluster.window_start:%H:%M:%S} → {cluster.window_end:%H:%M:%S}"
                    )
                    st.write(", ".join(cluster.ticket_ids))

                    assess_key = f"assess_{cluster.cluster_id}"

                    if st.button("🔍 Evalueaza", key=f"btn_{cluster.cluster_id}"):
                        with st.spinner("Assessment Agent evalueaza clusterul..."):
                            try:
                                st.session_state[assess_key] = assess_incident(cluster, summaries)
                            except AssessmentError as exc:
                                st.session_state[assess_key] = exc

                    if assess_key in st.session_state:
                        result = st.session_state[assess_key]
                        if isinstance(result, AssessmentError):
                            st.error(f"Eroare Assessment Agent: {result}")
                        else:
                            severity_color = {"SEV1": "🔴", "SEV2": "🟠", "SEV3": "🟡", "Unknown": "⚪"}
                            st.markdown(
                                f"{severity_color.get(result.estimated_severity, '⚪')} "
                                f"**{result.estimated_severity}** · "
                                f"candidate={result.is_major_incident_candidate} · "
                                f"confidence={result.confidence} · "
                                f"actiune: `{result.recommended_action}`"
                            )
                            with st.expander("Reasoning + surse RAG"):
                                st.write(result.reasoning)
                                st.caption(f"Surse: {', '.join(result.rag_sources)}")