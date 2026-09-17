# """
# UI (Streamlit) pentru demonstrarea fluxului MIA end-to-end.

# v1: simulare temporala peste tichetele mock reale + Detection Pipeline live
# (embeddings + clustering).

# v2: buton "Evalueaza cluster" -> Assessment Agent (Etapa 5).

# v3: human-in-the-loop (doc. sectiunea 10) - butoane Aproba/Respinge per
# cluster evaluat -> creeaza un MajorIncident, tinut in st.session_state.

# v4: dupa Aprobare, genereaza automat cele 2 CommunicationDraft (end_users +
# management) prin Communication Agent (Etapa 6), fiecare cu propriul buton
# de aprobare - human-in-the-loop separat, conform CommunicationDraft.requires_approval.

# v5 (acest fisier): redesign vizual - consola tehnica, fara emoji, paleta
# neutra + culori de severitate ca badge-uri functionale, tipografie Inter,
# margini subtiri in loc de carduri cu umbra. NICIO modificare de logica sau
# de chei din session_state fata de versiunea anterioara - doar stratul de
# prezentare.

# Rulare: `python -m streamlit run app/ui/streamlit_app.py`
# (NU `streamlit run ...` direct - vezi status, gotcha venv/launcher Windows).
# """

# from __future__ import annotations

# from datetime import datetime, timedelta, timezone

# import streamlit as st

# from app.agents.assessment_agent import AssessmentError, assess_incident
# from app.agents.communication_agent import CommunicationError, generate_communication
# from app.config import get_settings
# from app.detection.build_cluster import build_incident_cluster
# from app.detection.clustering import cluster_similar_tickets
# from app.detection.embeddings import create_embedding
# from app.detection.similarity import calculate_similarity_matrix
# from app.detection.time_window import filter_tickets_by_time_window
# from app.ingestion.mock_jira_api import fetch_tickets
# from app.models.schemas import MajorIncident

# st.set_page_config(page_title="Major Incident Agent", layout="wide")

# settings = get_settings()


# # ---------------------------------------------------------------------------
# # Stil vizual: consola tehnica, paleta neutra, tipografie Inter.
# # Selectorii data-testid tintesc structura interna curenta a Streamlit -
# # daca versiunea de Streamlit se schimba semnificativ, unele reguli pot
# # necesita ajustare, dar functionalitatea widget-urilor nu e afectata.
# # ---------------------------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#     }

#     :root {
#         --mia-bg: #F5F6F8;
#         --mia-surface: #FFFFFF;
#         --mia-border: #DBDFE5;
#         --mia-text: #1B2430;
#         --mia-text-muted: #667085;
#         --mia-accent: #2E5AAC;
#         --mia-sev1: #C0392B;
#         --mia-sev2: #C0791E;
#         --mia-sev3: #667085;
#         --mia-success: #2E7D46;
#     }

#     [data-testid="stAppViewContainer"] { background-color: var(--mia-bg); }
#     [data-testid="stHeader"] { background-color: transparent; }
#     #MainMenu, footer { visibility: hidden; }

#     h1, h2, h3 { color: var(--mia-text); font-weight: 600; letter-spacing: -0.01em; }

#     [data-testid="stCaptionContainer"] p,
#     [data-testid="stCaptionContainer"] { color: var(--mia-text-muted); }

#     div[data-testid="stVerticalBlockBorderWrapper"] {
#         border: 1px solid var(--mia-border) !important;
#         border-radius: 3px !important;
#         box-shadow: none !important;
#         background-color: var(--mia-surface);
#     }

#     .stButton button {
#         border-radius: 3px;
#         border: 1px solid var(--mia-border);
#         font-weight: 500;
#         color: var(--mia-text);
#     }
#     .stButton button:hover { border-color: var(--mia-accent); color: var(--mia-accent); }

#     [data-testid="stDataFrame"] { border: 1px solid var(--mia-border); border-radius: 3px; }

#     .mia-kicker {
#         font-size: 0.85rem;
#         color: var(--mia-text-muted);
#         margin-bottom: 0.15rem;
#     }

#     .mia-meta {
#         font-size: 0.85rem;
#         color: var(--mia-text-muted);
#         line-height: 1.6;
#     }

#     .mia-mono {
#         font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
#         font-size: 0.82rem;
#         color: var(--mia-text);
#     }

#     .mia-cluster-title {
#         display: flex;
#         align-items: baseline;
#         gap: 0.6rem;
#         font-size: 1rem;
#     }
#     .mia-cluster-title .id { font-weight: 600; }
#     .mia-cluster-title .service { color: var(--mia-text-muted); }

#     .mia-badge {
#         display: inline-block;
#         padding: 0.12rem 0.5rem;
#         border-radius: 3px;
#         font-size: 0.76rem;
#         font-weight: 600;
#         color: #FFFFFF;
#     }
#     .mia-badge.sev1 { background-color: var(--mia-sev1); }
#     .mia-badge.sev2 { background-color: var(--mia-sev2); }
#     .mia-badge.sev3 { background-color: var(--mia-sev3); }
#     .mia-badge.unknown { background-color: #9AA1AC; }

#     .mia-status {
#         display: inline-block;
#         padding: 0.12rem 0.5rem;
#         border-radius: 3px;
#         font-size: 0.76rem;
#         font-weight: 500;
#     }
#     .mia-status.declared { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }
#     .mia-status.rejected { background-color: rgba(102,112,133,0.14); color: var(--mia-text-muted); }
#     .mia-status.approved { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }

#     .mia-comm-label {
#         font-size: 0.78rem;
#         font-weight: 600;
#         color: var(--mia-text-muted);
#         margin-bottom: 0.2rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# def _severity_badge(severity: str) -> str:
#     css_class = {"SEV1": "sev1", "SEV2": "sev2", "SEV3": "sev3"}.get(severity, "unknown")
#     return f'<span class="mia-badge {css_class}">{severity}</span>'


# @st.cache_resource(show_spinner="Se incarca tichetele mock...")
# def load_all_tickets() -> list[dict]:
#     """
#     Incarca o singura data toate tichetele mock (nu se schimba intre rulari).
#     limit=500: maximul acceptat de mock_jira_api.py (Query(..., le=500)).
#     """
#     return fetch_tickets(limit=500)["issues"]


# @st.cache_data(show_spinner=False)
# def cached_embedding(text: str) -> list[float]:
#     """
#     Wrapper cache peste create_embedding() (Ollama). Cheia de cache e chiar
#     textul - daca acelasi tichet apare in ferestre diferite (slider miscat),
#     nu se recalculeaza embedding-ul de fiecare data.
#     """
#     return create_embedding(text)


# def _ticket_created_at(ticket: dict) -> datetime:
#     value = ticket["fields"]["created"]
#     if value.endswith("Z"):
#         value = value[:-1] + "+00:00"
#     if value.endswith("+0000"):
#         value = value[:-5] + "+00:00"
#     return datetime.fromisoformat(value)


# def _ticket_service(ticket: dict) -> str:
#     components = ticket["fields"].get("components") or []
#     return components[0]["name"] if components else "Unknown"


# all_tickets = load_all_tickets()
# all_created = [_ticket_created_at(t) for t in all_tickets]
# min_dt_naive = min(all_created).replace(tzinfo=None)
# max_dt_naive = max(all_created).replace(tzinfo=None)

# st.markdown('<div class="mia-kicker">Consola operare incidente</div>', unsafe_allow_html=True)
# st.title("Major Incident Agent")
# st.caption(
#     f"Simulare peste {len(all_tickets)} tichete mock "
#     f"({min_dt_naive:%Y-%m-%d %H:%M} – {max_dt_naive:%Y-%m-%d %H:%M} UTC). "
#     f"Fereastra de corelare: {settings.clustering_window_minutes} min."
# )

# selected_naive = st.slider(
#     "Timp simulat (momentul curent al sistemului)",
#     min_value=min_dt_naive,
#     max_value=max_dt_naive,
#     value=min_dt_naive + timedelta(hours=14, minutes=21),  # ~ burst GT-001 (Internal Portal)
#     step=timedelta(minutes=5),
#     format="DD/MM HH:mm",
# )
# reference_time = selected_naive.replace(tzinfo=timezone.utc)

# window_tickets = filter_tickets_by_time_window(
#     tickets=all_tickets,
#     reference_time=reference_time,
#     window_minutes=settings.clustering_window_minutes,
# )

# col_left, col_right = st.columns([1, 1])

# with col_left:
#     st.subheader(f"Tichete in fereastra curenta ({len(window_tickets)})")
#     if not window_tickets:
#         st.caption("Niciun tichet in aceasta fereastra de timp.")
#     else:
#         table_rows = [
#             {
#                 "ID": t["key"],
#                 "Ora": _ticket_created_at(t).strftime("%H:%M:%S"),
#                 "Serviciu": _ticket_service(t),
#                 "Rezumat": t["fields"]["summary"],
#             }
#             for t in sorted(window_tickets, key=_ticket_created_at)
#         ]
#         st.dataframe(table_rows, hide_index=True, use_container_width=True)

# with col_right:
#     st.subheader("Clustere candidate detectate")

#     if len(window_tickets) < settings.min_tickets_per_cluster:
#         st.caption(f"Sub minimul de {settings.min_tickets_per_cluster} tichete — nu se ruleaza clustering.")
#     else:
#         with st.spinner("Se calculeaza embeddings + similaritate..."):
#             texts = [t["fields"]["summary"] for t in window_tickets]
#             embeddings = [cached_embedding(text) for text in texts]
#             similarity_matrix = calculate_similarity_matrix(embeddings)
#             raw_clusters = cluster_similar_tickets(
#                 similarity_matrix,
#                 threshold=settings.similarity_threshold,
#                 min_cluster_size=settings.min_tickets_per_cluster,
#             )

#         if not raw_clusters:
#             st.caption("Niciun cluster peste prag in aceasta fereastra.")
#         else:
#             for seq, indices in enumerate(raw_clusters, start=1):
#                 cluster = build_incident_cluster(window_tickets, similarity_matrix, indices, seq)
#                 cluster_tickets = [window_tickets[i] for i in indices]
#                 summaries = [t["fields"]["summary"] for t in cluster_tickets]

#                 with st.container(border=True):
#                     st.markdown(
#                         f'<div class="mia-cluster-title">'
#                         f'<span class="id mia-mono">{cluster.cluster_id}</span>'
#                         f'<span class="service">{cluster.service_guess}</span>'
#                         f'</div>',
#                         unsafe_allow_html=True,
#                     )
#                     st.markdown(
#                         f'<div class="mia-meta">'
#                         f'{cluster.ticket_count} tichete &nbsp;|&nbsp; '
#                         f'similaritate {cluster.centroid_similarity:.2f} &nbsp;|&nbsp; '
#                         f'{cluster.window_start:%H:%M:%S}–{cluster.window_end:%H:%M:%S}'
#                         f'</div>',
#                         unsafe_allow_html=True,
#                     )
#                     st.markdown(
#                         f'<div class="mia-mono" style="margin-top:0.3rem; word-break:break-all;">'
#                         f'{", ".join(cluster.ticket_ids)}</div>',
#                         unsafe_allow_html=True,
#                     )

#                     st.write("")
#                     assess_key = f"assess_{cluster.cluster_id}"

#                     if st.button("Evalueaza cluster", key=f"btn_{cluster.cluster_id}"):
#                         with st.spinner("Assessment Agent evalueaza clusterul..."):
#                             try:
#                                 st.session_state[assess_key] = assess_incident(cluster, summaries)
#                             except AssessmentError as exc:
#                                 st.session_state[assess_key] = exc

#                     if assess_key in st.session_state:
#                         result = st.session_state[assess_key]
#                         if isinstance(result, AssessmentError):
#                             st.error(f"Eroare Assessment Agent: {result}")
#                         else:
#                             st.markdown(
#                                 f'<div class="mia-meta" style="margin-top:0.5rem;">'
#                                 f'{_severity_badge(result.estimated_severity)}'
#                                 f'&nbsp;&nbsp;Candidat Major Incident: <strong>'
#                                 f'{"da" if result.is_major_incident_candidate else "nu"}</strong>'
#                                 f'&nbsp;&nbsp;Confidence: <strong>{result.confidence:.2f}</strong>'
#                                 f'&nbsp;&nbsp;Actiune recomandata: '
#                                 f'<span class="mia-mono">{result.recommended_action}</span>'
#                                 f'</div>',
#                                 unsafe_allow_html=True,
#                             )
#                             with st.expander("Rationament si surse RAG"):
#                                 st.write(result.reasoning)
#                                 st.caption(f"Surse: {', '.join(result.rag_sources)}")

#                             decision_key = f"decision_{cluster.cluster_id}"

#                             if decision_key not in st.session_state:
#                                 col_a, col_b = st.columns(2)
#                                 with col_a:
#                                     if st.button("Aproba Major Incident", key=f"approve_{cluster.cluster_id}"):
#                                         st.session_state[decision_key] = MajorIncident(
#                                             incident_id=f"MI-{cluster.cluster_id}",
#                                             cluster_id=cluster.cluster_id,
#                                             status="Declared",
#                                             severity=result.estimated_severity,
#                                             declared_by="demo_user",
#                                             declared_at=datetime.now(timezone.utc),
#                                             root_cause_suspected=result.reasoning[:200],
#                                         )
#                                         st.session_state.setdefault("major_incidents", []).append(
#                                             st.session_state[decision_key]
#                                         )
#                                 with col_b:
#                                     if st.button("Respinge", key=f"reject_{cluster.cluster_id}"):
#                                         st.session_state[decision_key] = MajorIncident(
#                                             incident_id=f"MI-{cluster.cluster_id}",
#                                             cluster_id=cluster.cluster_id,
#                                             status="Rejected",
#                                             severity=result.estimated_severity,
#                                             declared_by="demo_user",
#                                             declared_at=datetime.now(timezone.utc),
#                                         )
#                                         st.session_state.setdefault("major_incidents", []).append(
#                                             st.session_state[decision_key]
#                                         )
#                             else:
#                                 decision = st.session_state[decision_key]
#                                 status_class = "declared" if decision.status == "Declared" else "rejected"
#                                 st.markdown(
#                                     f'<div class="mia-meta" style="margin-top:0.4rem;">'
#                                     f'<span class="mia-status {status_class}">{decision.status}</span>'
#                                     f'&nbsp;&nbsp;de {decision.declared_by} la '
#                                     f'{decision.declared_at:%H:%M:%S}'
#                                     f'</div>',
#                                     unsafe_allow_html=True,
#                                 )

#                                 # v4 - dupa aprobare, genereaza automat comunicarile (o data)
#                                 if decision.status == "Declared":
#                                     comm_key = f"comm_{cluster.cluster_id}"

#                                     if comm_key not in st.session_state:
#                                         with st.spinner("Communication Agent genereaza draft-urile..."):
#                                             drafts: dict[str, object] = {}
#                                             for audience in ("end_users", "management"):
#                                                 try:
#                                                     drafts[audience] = generate_communication(
#                                                         decision, result, cluster, audience
#                                                     )
#                                                 except CommunicationError as exc:
#                                                     drafts[audience] = exc
#                                             st.session_state[comm_key] = drafts

#                                     st.markdown(
#                                         '<div style="margin-top:0.8rem; font-weight:600;">'
#                                         "Comunicari generate</div>",
#                                         unsafe_allow_html=True,
#                                     )
#                                     for audience, draft in st.session_state[comm_key].items():
#                                         label = "End users" if audience == "end_users" else "Management"
#                                         with st.container(border=True):
#                                             st.markdown(
#                                                 f'<div class="mia-comm-label">{label}</div>',
#                                                 unsafe_allow_html=True,
#                                             )
#                                             if isinstance(draft, CommunicationError):
#                                                 st.error(f"Eroare Communication Agent: {draft}")
#                                                 continue

#                                             st.markdown(f"**{draft.subject}**")
#                                             st.write(draft.body)
#                                             st.caption(f"Surse: {', '.join(draft.rag_sources)}")

#                                             comm_decision_key = f"comm_decision_{cluster.cluster_id}_{audience}"

#                                             if comm_decision_key not in st.session_state:
#                                                 if st.button(
#                                                     "Aproba comunicarea",
#                                                     key=f"approve_comm_{cluster.cluster_id}_{audience}",
#                                                 ):
#                                                     st.session_state[comm_decision_key] = "Approved"
#                                             else:
#                                                 st.markdown(
#                                                     '<span class="mia-status approved">Aprobata</span>',
#                                                     unsafe_allow_html=True,
#                                                 )

# st.divider()
# st.subheader("Istoric decizii (sesiunea curenta)")
# history = st.session_state.get("major_incidents", [])
# if not history:
#     st.caption("Nicio decizie inregistrata inca.")
# else:
#     history_rows = [
#         {
#             "Incident": mi.incident_id,
#             "Status": mi.status,
#             "Severitate": mi.severity,
#             "Decis de": mi.declared_by,
#             "Data/ora (UTC)": mi.declared_at.strftime("%Y-%m-%d %H:%M:%S"),
#         }
#         for mi in reversed(history)
#     ]
#     st.dataframe(history_rows, hide_index=True, use_container_width=True)

"""
UI (Streamlit) pentru demonstrarea fluxului MIA end-to-end.

v1: simulare temporala peste tichetele mock reale + Detection Pipeline live
(embeddings + clustering).

v2: buton "Evalueaza cluster" -> Assessment Agent (Etapa 5).

v3: human-in-the-loop (doc. sectiunea 10) - butoane Aproba/Respinge per
cluster evaluat -> creeaza un MajorIncident, tinut in st.session_state.

v4: dupa Aprobare, genereaza automat cele 2 CommunicationDraft (end_users +
management) prin Communication Agent (Etapa 6), fiecare cu propriul buton
de aprobare - human-in-the-loop separat, conform CommunicationDraft.requires_approval.

v5: redesign vizual - consola tehnica, fara emoji, paleta neutra + culori de
severitate ca badge-uri functionale, tipografie Inter, margini subtiri in
loc de carduri cu umbra. NICIO modificare de logica sau de chei din
session_state fata de versiunea anterioara - doar stratul de prezentare.

v6 (acest fisier): selector "Data source" - pe langa simularea live (slider
peste tot setul de 330 tichete), adauga scenarii demo curate
(DEMO_MAJOR_INCIDENTS din mock_jira_api.py), identificate manual ca fiind
Major Incident real (multiple formulari diferite ale aceleiasi cauze,
prioritate P1/P2, multi-locatie, fereastra stransa) - utile pentru a arata
exemple pozitive fara sa depinzi de gasirea manuala a ferestrei prin slider.
Restul fluxului (clustering, evaluare, aprobare, comunicari) e neschimbat -
primeste window_tickets la fel, indiferent de sursa aleasa.

Rulare: `python -m streamlit run app/ui/streamlit_app.py`
(NU `streamlit run ...` direct - vezi status, gotcha venv/launcher Windows).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from app.agents.assessment_agent import AssessmentError, assess_incident
from app.agents.communication_agent import CommunicationError, generate_communication
from app.config import get_settings
from app.detection.build_cluster import build_incident_cluster
from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.detection.time_window import filter_tickets_by_time_window
from app.ingestion.mock_jira_api import DEMO_MAJOR_INCIDENTS, fetch_tickets
from app.models.schemas import MajorIncident

st.set_page_config(page_title="Major Incident Agent", layout="wide")

settings = get_settings()


# ---------------------------------------------------------------------------
# Stil vizual: consola tehnica, paleta neutra, tipografie Inter.
# Selectorii data-testid tintesc structura interna curenta a Streamlit -
# daca versiunea de Streamlit se schimba semnificativ, unele reguli pot
# necesita ajustare, dar functionalitatea widget-urilor nu e afectata.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    :root {
        --mia-bg: #F5F6F8;
        --mia-surface: #FFFFFF;
        --mia-border: #DBDFE5;
        --mia-text: #1B2430;
        --mia-text-muted: #667085;
        --mia-accent: #2E5AAC;
        --mia-sev1: #C0392B;
        --mia-sev2: #C0791E;
        --mia-sev3: #667085;
        --mia-success: #2E7D46;
    }

    [data-testid="stAppViewContainer"] { background-color: var(--mia-bg); }
    [data-testid="stHeader"] { background-color: transparent; }
    #MainMenu, footer { visibility: hidden; }

    h1, h2, h3 { color: var(--mia-text); font-weight: 600; letter-spacing: -0.01em; }

    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] { color: var(--mia-text-muted); }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--mia-border) !important;
        border-radius: 3px !important;
        box-shadow: none !important;
        background-color: var(--mia-surface);
    }

    .stButton button {
        border-radius: 3px;
        border: 1px solid var(--mia-border);
        font-weight: 500;
        color: var(--mia-text);
    }
    .stButton button:hover { border-color: var(--mia-accent); color: var(--mia-accent); }

    [data-testid="stDataFrame"] { border: 1px solid var(--mia-border); border-radius: 3px; }

    .mia-kicker {
        font-size: 0.85rem;
        color: var(--mia-text-muted);
        margin-bottom: 0.15rem;
    }

    .mia-meta {
        font-size: 0.85rem;
        color: var(--mia-text-muted);
        line-height: 1.6;
    }

    .mia-mono {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.82rem;
        color: var(--mia-text);
    }

    .mia-cluster-title {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        font-size: 1rem;
    }
    .mia-cluster-title .id { font-weight: 600; }
    .mia-cluster-title .service { color: var(--mia-text-muted); }

    .mia-badge {
        display: inline-block;
        padding: 0.12rem 0.5rem;
        border-radius: 3px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    .mia-badge.sev1 { background-color: var(--mia-sev1); }
    .mia-badge.sev2 { background-color: var(--mia-sev2); }
    .mia-badge.sev3 { background-color: var(--mia-sev3); }
    .mia-badge.unknown { background-color: #9AA1AC; }

    .mia-status {
        display: inline-block;
        padding: 0.12rem 0.5rem;
        border-radius: 3px;
        font-size: 0.76rem;
        font-weight: 500;
    }
    .mia-status.declared { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }
    .mia-status.rejected { background-color: rgba(102,112,133,0.14); color: var(--mia-text-muted); }
    .mia-status.approved { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }

    .mia-comm-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--mia-text-muted);
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _severity_badge(severity: str) -> str:
    css_class = {"SEV1": "sev1", "SEV2": "sev2", "SEV3": "sev3"}.get(severity, "unknown")
    return f'<span class="mia-badge {css_class}">{severity}</span>'


@st.cache_resource(show_spinner="Se incarca tichetele mock...")
def load_all_tickets() -> list[dict]:
    """
    Incarca o singura data toate tichetele mock (nu se schimba intre rulari).
    limit=500: maximul acceptat de mock_jira_api.py (Query(..., le=500)).
    """
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


# Etichete afisate in selector pentru scenariile demo, mapate pe cheile din
# DEMO_MAJOR_INCIDENTS (mock_jira_api.py). Textul e in engleza, coerent cu
# numele scenariilor definite in cod.
# Etichete afisate in selector pentru scenariile demo, mapate pe cheile din
# DEMO_MAJOR_INCIDENTS (mock_jira_api.py).
DEMO_LABELS: dict[str, str] = {
    key: f"Demo: {key.replace('_', ' ').title()}"
    for key in DEMO_MAJOR_INCIDENTS
}
_LABEL_TO_KEY = {v: k for k, v in DEMO_LABELS.items()}

all_tickets = load_all_tickets()
all_created = [_ticket_created_at(t) for t in all_tickets]
min_dt_naive = min(all_created).replace(tzinfo=None)
max_dt_naive = max(all_created).replace(tzinfo=None)

# st.markdown('<div class="mia-kicker">Consola operare incidente</div>', unsafe_allow_html=True)
st.title("Major Incident Agent")
st.caption(
    f"Simulare peste {len(all_tickets)} tichete mock "
    f"({min_dt_naive:%Y-%m-%d %H:%M} – {max_dt_naive:%Y-%m-%d %H:%M} UTC). "
    f"Fereastra de corelare: {settings.clustering_window_minutes} min."
)

data_source = st.radio(
    "Data source",
    options=["Live simulation"] + list(DEMO_LABELS.values()),
    horizontal=True,
)

if data_source == "Live simulation":
    selected_naive = st.slider(
        "Timp simulat (momentul curent al sistemului)",
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
else:
    demo_key = _LABEL_TO_KEY[data_source]
    window_tickets = fetch_tickets(ticket_keys=DEMO_MAJOR_INCIDENTS[demo_key])["issues"]
    st.caption(f"{len(window_tickets)} tichete incarcate din scenariul demo curat.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"Tichete in fereastra curenta ({len(window_tickets)})")
    if not window_tickets:
        st.caption("Niciun tichet in aceasta fereastra de timp.")
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
    st.subheader("Clustere candidate detectate")

    if len(window_tickets) < settings.min_tickets_per_cluster:
        st.caption(f"Sub minimul de {settings.min_tickets_per_cluster} tichete — nu se ruleaza clustering.")
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
            st.caption("Niciun cluster peste prag in aceasta fereastra.")
        else:
            for seq, indices in enumerate(raw_clusters, start=1):
                cluster = build_incident_cluster(window_tickets, similarity_matrix, indices, seq)
                cluster_tickets = [window_tickets[i] for i in indices]
                summaries = [t["fields"]["summary"] for t in cluster_tickets]

                with st.container(border=True):
                    st.markdown(
                        f'<div class="mia-cluster-title">'
                        f'<span class="id mia-mono">{cluster.cluster_id}</span>'
                        f'<span class="service">{cluster.service_guess}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="mia-meta">'
                        f'{cluster.ticket_count} tichete &nbsp;|&nbsp; '
                        f'similaritate {cluster.centroid_similarity:.2f} &nbsp;|&nbsp; '
                        f'{cluster.window_start:%H:%M:%S}–{cluster.window_end:%H:%M:%S}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="mia-mono" style="margin-top:0.3rem; word-break:break-all;">'
                        f'{", ".join(cluster.ticket_ids)}</div>',
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    assess_key = f"assess_{cluster.cluster_id}"

                    if st.button("Evalueaza cluster", key=f"btn_{cluster.cluster_id}"):
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
                            st.markdown(
                                f'<div class="mia-meta" style="margin-top:0.5rem;">'
                                f'{_severity_badge(result.estimated_severity)}'
                                f'&nbsp;&nbsp;Candidat Major Incident: <strong>'
                                f'{"da" if result.is_major_incident_candidate else "nu"}</strong>'
                                f'&nbsp;&nbsp;Confidence: <strong>{result.confidence:.2f}</strong>'
                                f'&nbsp;&nbsp;Actiune recomandata: '
                                f'<span class="mia-mono">{result.recommended_action}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            with st.expander("Rationament si surse RAG"):
                                st.write(result.reasoning)
                                st.caption(f"Surse: {', '.join(result.rag_sources)}")

                            decision_key = f"decision_{cluster.cluster_id}"

                            if decision_key not in st.session_state:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("Aproba Major Incident", key=f"approve_{cluster.cluster_id}"):
                                        st.session_state[decision_key] = MajorIncident(
                                            incident_id=f"MI-{cluster.cluster_id}",
                                            cluster_id=cluster.cluster_id,
                                            status="Declared",
                                            severity=result.estimated_severity,
                                            declared_by="demo_user",
                                            declared_at=datetime.now(timezone.utc),
                                            root_cause_suspected=result.reasoning[:200],
                                        )
                                        st.session_state.setdefault("major_incidents", []).append(
                                            st.session_state[decision_key]
                                        )
                                with col_b:
                                    if st.button("Respinge", key=f"reject_{cluster.cluster_id}"):
                                        st.session_state[decision_key] = MajorIncident(
                                            incident_id=f"MI-{cluster.cluster_id}",
                                            cluster_id=cluster.cluster_id,
                                            status="Rejected",
                                            severity=result.estimated_severity,
                                            declared_by="demo_user",
                                            declared_at=datetime.now(timezone.utc),
                                        )
                                        st.session_state.setdefault("major_incidents", []).append(
                                            st.session_state[decision_key]
                                        )
                            else:
                                decision = st.session_state[decision_key]
                                status_class = "declared" if decision.status == "Declared" else "rejected"
                                st.markdown(
                                    f'<div class="mia-meta" style="margin-top:0.4rem;">'
                                    f'<span class="mia-status {status_class}">{decision.status}</span>'
                                    f'&nbsp;&nbsp;de {decision.declared_by} la '
                                    f'{decision.declared_at:%H:%M:%S}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                                # v4 - dupa aprobare, genereaza automat comunicarile (o data)
                                if decision.status == "Declared":
                                    comm_key = f"comm_{cluster.cluster_id}"

                                    if comm_key not in st.session_state:
                                        with st.spinner("Communication Agent genereaza draft-urile..."):
                                            drafts: dict[str, object] = {}
                                            for audience in ("end_users", "management"):
                                                try:
                                                    drafts[audience] = generate_communication(
                                                        decision, result, cluster, audience
                                                    )
                                                except CommunicationError as exc:
                                                    drafts[audience] = exc
                                            st.session_state[comm_key] = drafts

                                    st.markdown(
                                        '<div style="margin-top:0.8rem; font-weight:600;">'
                                        "Comunicari generate</div>",
                                        unsafe_allow_html=True,
                                    )
                                    for audience, draft in st.session_state[comm_key].items():
                                        label = "End users" if audience == "end_users" else "Management"
                                        with st.container(border=True):
                                            st.markdown(
                                                f'<div class="mia-comm-label">{label}</div>',
                                                unsafe_allow_html=True,
                                            )
                                            if isinstance(draft, CommunicationError):
                                                st.error(f"Eroare Communication Agent: {draft}")
                                                continue

                                            st.markdown(f"**{draft.subject}**")
                                            st.write(draft.body)
                                            st.caption(f"Surse: {', '.join(draft.rag_sources)}")

                                            comm_decision_key = f"comm_decision_{cluster.cluster_id}_{audience}"

                                            if comm_decision_key not in st.session_state:
                                                if st.button(
                                                    "Aproba comunicarea",
                                                    key=f"approve_comm_{cluster.cluster_id}_{audience}",
                                                ):
                                                    st.session_state[comm_decision_key] = "Approved"
                                            else:
                                                st.markdown(
                                                    '<span class="mia-status approved">Aprobata</span>',
                                                    unsafe_allow_html=True,
                                                )

st.divider()
st.subheader("Istoric decizii (sesiunea curenta)")
history = st.session_state.get("major_incidents", [])
if not history:
    st.caption("Nicio decizie inregistrata inca.")
else:
    history_rows = [
        {
            "Incident": mi.incident_id,
            "Status": mi.status,
            "Severitate": mi.severity,
            "Decis de": mi.declared_by,
            "Data/ora (UTC)": mi.declared_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for mi in reversed(history)
    ]
    st.dataframe(history_rows, hide_index=True, use_container_width=True)