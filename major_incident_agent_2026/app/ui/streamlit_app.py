"""
UI (Streamlit) pentru demonstrarea fluxului MIA end-to-end - REDESIGN v7.

Restructurare completa fata de versiunile anterioare, la cererea explicita a
stakeholder-ului (principii UI/UX inspirate din Material Design / MUI
stepper - https://mui.com/material-ui/all-components/):

1. Flux tip STEPPER: 6 pasi liniari, fiecare cu: (a) explicatie clara a
   nodului agentic care ruleaza si ce face, (b) buton de rulare a nodului,
   (c) afisarea rezultatului primit, (d) navigare Next/Back catre pasul
   urmator - NU mai multe clustere/carduri afisate simultan, fara flux clar.
2. Eliminat sliderul liber pe tot setul de 330 tichete si dropdown-ul de
   preset ca elemente disparate - inlocuite cu o singura selectie de scenariu
   in Pasul 1, urmata de o VIZUALIZARE LIVE a sosirii tichetelor (populate
   treptat intr-un tabel, cu delay), nu instantaneu.
3. Design profesional pastrat pe tot parcursul (paleta neutra, tipografie
   Inter, badge-uri functionale pentru severitate/status, fara emoji) -
   mostenit din v5, extins acum cu un header de tip stepper (indicator de
   progres pe 6 pasi).

NOTA (limitare Streamlit vs. MUI real): MUI e o libraria React - nu poate fi
folosita literal intr-o aplicatie Streamlit (Python). Acest fisier
reproduce vizual principiile stepper-ului Material (indicator de progres,
un pas activ, continut clar per pas), construit manual din CSS/HTML, nu
importat ca atare. Discutat si confirmat cu userul ca varianta aleasa
(varianta A din cele 3 propuse).

Rulare: `python -m streamlit run app/ui/streamlit_app.py`
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import streamlit as st

from app.agents.assessment_agent import AssessmentError, assess_incident
from app.agents.communication_agent import CommunicationError, generate_communication
from app.config import get_settings
from app.detection.build_cluster import build_incident_cluster
from app.detection.clustering import cluster_similar_tickets
from app.detection.embeddings import create_embedding
from app.detection.similarity import calculate_similarity_matrix
from app.ingestion.mock_jira_api import DEMO_MAJOR_INCIDENTS, fetch_tickets
from app.models.schemas import MajorIncident

st.set_page_config(page_title="Major Incident Agent", layout="wide")

settings = get_settings()


# ---------------------------------------------------------------------------
# Stil vizual: consola tehnica, paleta neutra, tipografie Inter, plus header
# de tip stepper (indicator de progres pe 6 pasi).
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
        --mia-accent-soft: #E4EAF5;
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

    .mia-kicker { font-size: 0.85rem; color: var(--mia-text-muted); margin-bottom: 0.15rem; }
    .mia-meta { font-size: 0.85rem; color: var(--mia-text-muted); line-height: 1.6; }
    .mia-mono {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.82rem; color: var(--mia-text);
    }
    .mia-cluster-title { display: flex; align-items: baseline; gap: 0.6rem; font-size: 1rem; }
    .mia-cluster-title .id { font-weight: 600; }
    .mia-cluster-title .service { color: var(--mia-text-muted); }

    .mia-badge {
        display: inline-block; padding: 0.12rem 0.5rem; border-radius: 3px;
        font-size: 0.76rem; font-weight: 600; color: #FFFFFF;
    }
    .mia-badge.sev1 { background-color: var(--mia-sev1); }
    .mia-badge.sev2 { background-color: var(--mia-sev2); }
    .mia-badge.sev3 { background-color: var(--mia-sev3); }
    .mia-badge.unknown { background-color: #9AA1AC; }

    .mia-status {
        display: inline-block; padding: 0.12rem 0.5rem; border-radius: 3px;
        font-size: 0.76rem; font-weight: 500;
    }
    .mia-status.declared { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }
    .mia-status.rejected { background-color: rgba(102,112,133,0.14); color: var(--mia-text-muted); }
    .mia-status.approved { background-color: rgba(46,125,70,0.12); color: var(--mia-success); }

    .mia-comm-label { font-size: 0.78rem; font-weight: 600; color: var(--mia-text-muted); margin-bottom: 0.2rem; }

    .mia-node-box {
        background-color: var(--mia-accent-soft);
        border: 1px solid var(--mia-accent);
        border-radius: 3px;
        padding: 0.7rem 0.9rem;
        font-size: 0.88rem;
        color: var(--mia-text);
        margin-bottom: 0.9rem;
    }
    .mia-node-box .node-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--mia-accent);
        margin-bottom: 0.2rem;
    }

    /* Stepper header */
    .mia-stepper { display: flex; align-items: flex-start; margin: 1.2rem 0 1.6rem 0; }
    .mia-step { flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; }
    .mia-step:not(:last-child)::after {
        content: ""; position: absolute; top: 14px; left: 50%; width: 100%; height: 2px;
        background-color: var(--mia-border); z-index: 0;
    }
    .mia-step.done:not(:last-child)::after { background-color: var(--mia-accent); }
    .mia-step-circle {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 600; z-index: 1;
        background-color: var(--mia-surface); border: 2px solid var(--mia-border);
        color: var(--mia-text-muted);
    }
    .mia-step.done .mia-step-circle { background-color: var(--mia-accent); border-color: var(--mia-accent); color: #FFFFFF; }
    .mia-step.current .mia-step-circle { border-color: var(--mia-accent); color: var(--mia-accent); }
    .mia-step-label {
        font-size: 0.74rem; color: var(--mia-text-muted); margin-top: 0.35rem; text-align: center; max-width: 110px;
    }
    .mia-step.current .mia-step-label { color: var(--mia-text); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _severity_badge(severity: str) -> str:
    css_class = {"SEV1": "sev1", "SEV2": "sev2", "SEV3": "sev3"}.get(severity, "unknown")
    return f'<span class="mia-badge {css_class}">{severity}</span>'


def _node_box(node_name: str, description: str) -> None:
    """Afiseaza explicit ce nod agentic ruleaza la acest pas si ce face."""
    st.markdown(
        f'<div class="mia-node-box">'
        f'<div class="node-label">Node: {node_name}</div>'
        f'{description}'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_embedding(text: str) -> list[float]:
    """Wrapper cache peste create_embedding() (Ollama)."""
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


def _tickets_table(tickets: list[dict]) -> list[dict]:
    return [
        {
            "ID": t["key"],
            "Ora": _ticket_created_at(t).strftime("%H:%M:%S"),
            "Serviciu": _ticket_service(t),
            "Rezumat": t["fields"]["summary"],
        }
        for t in sorted(tickets, key=_ticket_created_at)
    ]


# ---------------------------------------------------------------------------
# Definitia pasilor stepper-ului
# ---------------------------------------------------------------------------
STEPS = [
    "Sosire tichete",
    "Detectie & clustering",
    "Evaluare AI (LLM + RAG)",
    "Aprobare umana",
    "Comunicari",
    "Sumar",
]

DEMO_LABELS: dict[str, str] = {
    key: f"Demo: {key.replace('_', ' ').title()}" for key in DEMO_MAJOR_INCIDENTS
}


def _init_state() -> None:
    defaults = {
        "wizard_step": 0,
        "scenario_key": None,
        "stream_tickets": [],
        "stream_done": False,
        "detection_result": None,
        "selected_cluster_id": None,
        "assessment_result": None,
        "decision": None,
        "comm_drafts": None,
        "comm_decisions": {},
        "major_incidents": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_flow() -> None:
    """Reseteaza flow-ul curent (dar pastreaza istoricul de decizii)."""
    for key in (
        "scenario_key", "stream_tickets", "stream_done", "detection_result",
        "selected_cluster_id", "assessment_result", "decision",
        "comm_drafts", "comm_decisions",
    ):
        st.session_state[key] = [] if key in ("stream_tickets",) else (
            {} if key == "comm_decisions" else None if key != "stream_done" else False
        )
    st.session_state["wizard_step"] = 0


_init_state()


def _render_stepper_header(current: int) -> None:
    cells = []
    for i, label in enumerate(STEPS):
        state = "done" if i < current else ("current" if i == current else "")
        marker = "\u2713" if i < current else str(i + 1)
        cells.append(
            f'<div class="mia-step {state}">'
            f'<div class="mia-step-circle">{marker}</div>'
            f'<div class="mia-step-label">{label}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="mia-stepper">{"".join(cells)}</div>', unsafe_allow_html=True)


st.markdown('<div class="mia-kicker">Consola operare incidente</div>', unsafe_allow_html=True)
st.title("Major Incident Agent")
st.caption("Flux agentic pas cu pas: detectie, evaluare AI, aprobare umana, comunicare.")

_render_stepper_header(st.session_state["wizard_step"])

step = st.session_state["wizard_step"]

# ===========================================================================
# PASUL 1 - Sosire tichete (live)
# ===========================================================================
if step == 0:
    st.subheader(STEPS[0])
    _node_box(
        "Ticket Intake",
        "Simuleaza sosirea tichetelor in sistem, in ordine cronologica, "
        "exact cum ar fi ingerate dintr-un Jira real.",
    )

    if not st.session_state["stream_done"]:
        chosen_label = st.selectbox(
            "Alege un scenariu de incident pentru redare",
            options=list(DEMO_LABELS.values()),
        )
        chosen_key = [k for k, v in DEMO_LABELS.items() if v == chosen_label][0]

        if st.button("Porneste sosirea tichetelor"):
            st.session_state["scenario_key"] = chosen_key
            all_tickets = sorted(
                fetch_tickets(ticket_keys=DEMO_MAJOR_INCIDENTS[chosen_key])["issues"],
                key=_ticket_created_at,
            )
            placeholder = st.empty()
            status = st.empty()
            arrived: list[dict] = []
            for i, ticket in enumerate(all_tickets, start=1):
                arrived.append(ticket)
                status.caption(f"Se primeste tichetul {i} din {len(all_tickets)}...")
                placeholder.dataframe(_tickets_table(arrived), hide_index=True, use_container_width=True)
                time.sleep(0.35)
            status.caption(f"Toate cele {len(all_tickets)} tichete au fost primite.")
            st.session_state["stream_tickets"] = arrived
            st.session_state["stream_done"] = True
            st.rerun()
    else:
        st.caption(f"{len(st.session_state['stream_tickets'])} tichete primite din scenariul ales.")
        st.dataframe(_tickets_table(st.session_state["stream_tickets"]), hide_index=True, use_container_width=True)
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("Reia"):
                _reset_flow()
                st.rerun()
        with col_b:
            if st.button("Next: Detectie & clustering"):
                st.session_state["wizard_step"] = 1
                st.rerun()

# ===========================================================================
# PASUL 2 - Detectie & clustering
# ===========================================================================
elif step == 1:
    st.subheader(STEPS[1])
    _node_box(
        "Detection Pipeline",
        "Calculeaza embeddings BGE-M3 pentru fiecare tichet, similaritatea "
        "cosinus intre toate perechile, apoi grupeaza tichetele corelate "
        "(connected components, prag "
        f"{settings.similarity_threshold}, minim {settings.min_tickets_per_cluster} tichete/cluster).",
    )

    tickets = st.session_state["stream_tickets"]

    if st.session_state["detection_result"] is None:
        if st.button("Ruleaza detectia"):
            with st.spinner("Se calculeaza embeddings + similaritate..."):
                texts = [t["fields"]["summary"] for t in tickets]
                embeddings = [cached_embedding(text) for text in texts]
                similarity_matrix = calculate_similarity_matrix(embeddings)
                raw_clusters = cluster_similar_tickets(
                    similarity_matrix,
                    threshold=settings.similarity_threshold,
                    min_cluster_size=settings.min_tickets_per_cluster,
                )
                clusters = [
                    build_incident_cluster(tickets, similarity_matrix, idx, seq)
                    for seq, idx in enumerate(raw_clusters, start=1)
                ]
                st.session_state["detection_result"] = {
                    "tickets": tickets,
                    "clusters": clusters,
                    "indices_by_cluster": {c.cluster_id: idx for c, idx in zip(clusters, raw_clusters)},
                }
                if len(clusters) == 1:
                    st.session_state["selected_cluster_id"] = clusters[0].cluster_id
            st.rerun()
    else:
        clusters = st.session_state["detection_result"]["clusters"]

        if not clusters:
            st.warning("Niciun cluster peste prag in acest scenariu.")
            if st.button("Back"):
                st.session_state["wizard_step"] = 0
                st.rerun()
        else:
            if len(clusters) > 1:
                options = {c.cluster_id: c for c in clusters}
                chosen = st.radio(
                    "Mai multe clustere detectate - alege unul pentru a continua",
                    options=list(options.keys()),
                )
                st.session_state["selected_cluster_id"] = chosen

            selected = next(c for c in clusters if c.cluster_id == st.session_state["selected_cluster_id"])
            with st.container(border=True):
                st.markdown(
                    f'<div class="mia-cluster-title">'
                    f'<span class="id mia-mono">{selected.cluster_id}</span>'
                    f'<span class="service">{selected.service_guess}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mia-meta">'
                    f'{selected.ticket_count} tichete &nbsp;|&nbsp; '
                    f'similaritate {selected.centroid_similarity:.2f} &nbsp;|&nbsp; '
                    f'{selected.window_start:%H:%M:%S}\u2013{selected.window_end:%H:%M:%S}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mia-mono" style="margin-top:0.3rem; word-break:break-all;">'
                    f'{", ".join(selected.ticket_ids)}</div>',
                    unsafe_allow_html=True,
                )

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Back"):
                    st.session_state["wizard_step"] = 0
                    st.rerun()
            with col_b:
                if st.button("Next: Evaluare AI"):
                    st.session_state["wizard_step"] = 2
                    st.rerun()

# ===========================================================================
# PASUL 3 - Evaluare AI (LLM + RAG)
# ===========================================================================
elif step == 2:
    st.subheader(STEPS[2])
    _node_box(
        "Assessment Agent",
        "Interogheaza knowledge base-ul (incidente istorice similare si "
        "runbook-ul de severitate pentru acest serviciu, via RAG/ChromaDB), "
        f"apoi cere modelului LLM ({settings.ollama_llm_model}) sa clasifice "
        "severitatea si sa recomande o actiune.",
    )

    detection = st.session_state["detection_result"]
    cluster = next(c for c in detection["clusters"] if c.cluster_id == st.session_state["selected_cluster_id"])
    indices = detection["indices_by_cluster"][cluster.cluster_id]
    summaries = [detection["tickets"][i]["fields"]["summary"] for i in indices]

    if st.session_state["assessment_result"] is None:
        if st.button("Ruleaza evaluarea"):
            with st.spinner("Assessment Agent evalueaza clusterul..."):
                try:
                    st.session_state["assessment_result"] = assess_incident(cluster, summaries)
                except AssessmentError as exc:
                    st.session_state["assessment_result"] = exc
            st.rerun()
    else:
        result = st.session_state["assessment_result"]
        if isinstance(result, AssessmentError):
            st.error(f"Eroare Assessment Agent: {result}")
            if st.button("Reincearca"):
                st.session_state["assessment_result"] = None
                st.rerun()
        else:
            st.markdown(
                f'<div class="mia-meta">'
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

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Back"):
                    st.session_state["wizard_step"] = 1
                    st.rerun()
            with col_b:
                if st.button("Next: Aprobare umana"):
                    st.session_state["wizard_step"] = 3
                    st.rerun()

# ===========================================================================
# PASUL 4 - Aprobare umana (human-in-the-loop)
# ===========================================================================
elif step == 3:
    st.subheader(STEPS[3])
    _node_box(
        "Human-in-the-loop",
        "Declararea unui Major Incident necesita aprobare umana obligatorie "
        "(doc. sectiunea 10) - recomandarea AI-ului de mai sus e o propunere, "
        "nu o decizie automata.",
    )

    result = st.session_state["assessment_result"]
    cluster = next(
        c for c in st.session_state["detection_result"]["clusters"]
        if c.cluster_id == st.session_state["selected_cluster_id"]
    )

    st.markdown(
        f'<div class="mia-meta">Recomandare AI: {_severity_badge(result.estimated_severity)} '
        f'&nbsp;&nbsp;actiune: <span class="mia-mono">{result.recommended_action}</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state["decision"] is None:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Aproba Major Incident"):
                decision = MajorIncident(
                    incident_id=f"MI-{cluster.cluster_id}",
                    cluster_id=cluster.cluster_id,
                    status="Declared",
                    severity=result.estimated_severity,
                    declared_by="demo_user",
                    declared_at=datetime.now(timezone.utc),
                    root_cause_suspected=result.reasoning[:200],
                )
                st.session_state["decision"] = decision
                st.session_state["major_incidents"].append(decision)
                st.rerun()
        with col_b:
            if st.button("Respinge"):
                decision = MajorIncident(
                    incident_id=f"MI-{cluster.cluster_id}",
                    cluster_id=cluster.cluster_id,
                    status="Rejected",
                    severity=result.estimated_severity,
                    declared_by="demo_user",
                    declared_at=datetime.now(timezone.utc),
                )
                st.session_state["decision"] = decision
                st.session_state["major_incidents"].append(decision)
                st.rerun()
    else:
        decision = st.session_state["decision"]
        status_class = "declared" if decision.status == "Declared" else "rejected"
        st.markdown(
            f'<div class="mia-meta">'
            f'<span class="mia-status {status_class}">{decision.status}</span>'
            f'&nbsp;&nbsp;de {decision.declared_by} la {decision.declared_at:%H:%M:%S}'
            f'</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("Back"):
                st.session_state["wizard_step"] = 2
                st.rerun()
        with col_b:
            if st.button("Next: Comunicari"):
                st.session_state["wizard_step"] = 4
                st.rerun()

# ===========================================================================
# PASUL 5 - Comunicari
# ===========================================================================
elif step == 4:
    st.subheader(STEPS[4])

    decision = st.session_state["decision"]

    if decision.status == "Rejected":
        st.info("Incidentul a fost respins - nu se genereaza comunicari.")
        if st.button("Next: Sumar"):
            st.session_state["wizard_step"] = 5
            st.rerun()
    else:
        _node_box(
            "Communication Agent",
            "Interogheaza template-urile de comunicare relevante per audienta "
            "(RAG), apoi cere LLM-ului sa redacteze mesajele pentru end users "
            "si management.",
        )

        result = st.session_state["assessment_result"]
        cluster = next(
            c for c in st.session_state["detection_result"]["clusters"]
            if c.cluster_id == st.session_state["selected_cluster_id"]
        )

        if st.session_state["comm_drafts"] is None:
            if st.button("Genereaza comunicarile"):
                with st.spinner("Communication Agent genereaza draft-urile..."):
                    drafts: dict[str, object] = {}
                    for audience in ("end_users", "management"):
                        try:
                            drafts[audience] = generate_communication(decision, result, cluster, audience)
                        except CommunicationError as exc:
                            drafts[audience] = exc
                    st.session_state["comm_drafts"] = drafts
                st.rerun()
        else:
            for audience, draft in st.session_state["comm_drafts"].items():
                label = "End users" if audience == "end_users" else "Management"
                with st.container(border=True):
                    st.markdown(f'<div class="mia-comm-label">{label}</div>', unsafe_allow_html=True)
                    if isinstance(draft, CommunicationError):
                        st.error(f"Eroare Communication Agent: {draft}")
                        continue

                    st.markdown(f"**{draft.subject}**")
                    st.write(draft.body)
                    st.caption(f"Surse: {', '.join(draft.rag_sources)}")

                    if audience not in st.session_state["comm_decisions"]:
                        if st.button("Aproba comunicarea", key=f"approve_comm_{audience}"):
                            st.session_state["comm_decisions"][audience] = "Approved"
                            st.rerun()
                    else:
                        st.markdown('<span class="mia-status approved">Aprobata</span>', unsafe_allow_html=True)

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Back"):
                    st.session_state["wizard_step"] = 3
                    st.rerun()
            with col_b:
                if st.button("Next: Sumar"):
                    st.session_state["wizard_step"] = 5
                    st.rerun()

# ===========================================================================
# PASUL 6 - Sumar
# ===========================================================================
elif step == 5:
    st.subheader(STEPS[5])

    decision = st.session_state["decision"]
    result = st.session_state["assessment_result"]
    cluster = next(
        c for c in st.session_state["detection_result"]["clusters"]
        if c.cluster_id == st.session_state["selected_cluster_id"]
    )

    with st.container(border=True):
        st.markdown(f"**Scenariu:** {DEMO_LABELS[st.session_state['scenario_key']]}")
        st.markdown(f"**Cluster:** `{cluster.cluster_id}` \u2014 {cluster.service_guess} ({cluster.ticket_count} tichete)")
        st.markdown(
            f"**Evaluare AI:** {_severity_badge(result.estimated_severity)} "
            f"&nbsp;actiune recomandata: `{result.recommended_action}`",
            unsafe_allow_html=True,
        )
        status_class = "declared" if decision.status == "Declared" else "rejected"
        st.markdown(
            f'**Decizie umana:** <span class="mia-status {status_class}">{decision.status}</span>',
            unsafe_allow_html=True,
        )
        if decision.status == "Declared" and st.session_state["comm_drafts"]:
            approved = list(st.session_state["comm_decisions"].keys())
            st.markdown(f"**Comunicari aprobate:** {', '.join(approved) if approved else 'niciuna inca'}")

    if st.button("Incepe un incident nou"):
        _reset_flow()
        st.rerun()

# ---------------------------------------------------------------------------
# Istoric decizii - persistent, vizibil indiferent de pasul curent
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Istoric decizii (sesiunea curenta)")
history = st.session_state["major_incidents"]
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