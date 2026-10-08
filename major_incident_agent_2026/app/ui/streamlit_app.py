"""
UI (Streamlit) pentru demonstrarea fluxului MIA end-to-end - INTEGRAT CU LANGGRAPH SERVER.

Integrare cu LangGraph:
- Graful NU mai ruleaza in procesul Streamlit. Ruleaza pe serverul `langgraph dev`
  (http://127.0.0.1:2024), iar UI-ul il apeleaza prin `langgraph_sdk`.
- Astfel, tot ce faci in Streamlit apare live in LangGraph Studio (thread-ul curent).
- Starea vine de la server ca dict-uri JSON (nu obiecte Pydantic).
- Punctele de Human-in-the-Loop folosesc interrupt() + resume prin `command={"resume": ...}`.
- Istoricul deciziilor vine din audit.db (SQLite), nu din session_state: supravietuieste
  la "Incepe un incident nou" si la restartul aplicatiei.

Pornire (2 terminale, cu venv activ):
  1) langgraph dev
  2) python -m streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape

import httpx
import streamlit as st
from langgraph_sdk import get_sync_client

from app.config import get_settings
from app.detection.build_cluster import build_incident_cluster
from app.detection.embeddings import create_embedding
from app.detection.pipeline import detect_clusters_with_matrix
from app.detection.unclustered import find_unclustered
from app.execution import audit_store, ticket_reviews
from app.ingestion import ticket_store

st.set_page_config(page_title="Major Incident Agent", layout="wide")

settings = get_settings()

LANGGRAPH_URL = "http://127.0.0.1:2024"
GRAPH_NAME = "mia"  # cheia din langgraph.json -> "graphs"
MOCK_URL = f"http://127.0.0.1:{settings.mock_jira_port}"

ticket_store.init_db()

# Operatorul care ia deciziile (trimis la resume si salvat in audit)
DECIDED_BY = st.sidebar.text_input("Operator", value="demo_user").strip() or "demo_user"

VIEW_FLOW = "Flux incident"
VIEW_HISTORY = "Istoric incidente"
st.sidebar.divider()
VIEW = st.sidebar.radio("Navigare", [VIEW_FLOW, VIEW_HISTORY], key="nav_view")


# ---------------------------------------------------------------------------
# Client LangGraph Server
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    return get_sync_client(url=LANGGRAPH_URL, timeout=600)


client = get_client()


def _run_graph(input_payload: dict | None = None, resume: dict | None = None) -> None:
    """Porneste un run nou (input) sau reia dupa interrupt (resume), pe thread-ul curent."""
    thread_id = st.session_state["thread_id"]
    try:
        if resume is not None:
            client.runs.wait(thread_id, GRAPH_NAME, command={"resume": resume})
        else:
            client.runs.wait(thread_id, GRAPH_NAME, input=input_payload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Eroare la apelul serverului LangGraph: {exc}")
        st.stop()


def _get_thread_state() -> dict:
    """Intoarce {"values": {...}, "next": [...]} de la server pentru thread-ul curent."""
    thread_id = st.session_state["thread_id"]
    if not thread_id:
        return {"values": {}, "next": []}
    try:
        return client.threads.get_state(thread_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nu pot citi starea thread-ului: {exc}")
        st.stop()


# Starile in care un cluster e considerat finalizat (se poate vedea doar rezultatul)
DONE_CODES = {"declared", "rejected", "not_candidate"}
# Pasul la care se redeschide un cluster, in functie de status
OPEN_STEP = {"pending": 2, "awaiting": 3, "comms": 4, "declared": 5, "rejected": 5, "not_candidate": 5}


def _cluster_status(thread_id: str | None) -> tuple[str, str]:
    """(cod, eticheta) pentru un cluster, calculat din starea thread-ului de pe server."""
    if not thread_id:
        return "new", "neevaluat"
    try:
        state = client.threads.get_state(thread_id)
    except Exception:  # noqa: BLE001
        return "pending", "stare indisponibilă"
    values = state.get("values") or {}
    if not values.get("assessment"):
        return "pending", "în evaluare"
    approved = values.get("user_approved_incident")
    ai_candidate = values["assessment"].get("is_major_incident_candidate", True)
    if approved is True:
        suffix = "" if ai_candidate else " (nepropus de AI)"
        if values.get("user_approved_communications"):
            return "declared", f"declarat{suffix}"
        return "comms", f"declarat{suffix}, comunicări neaprobate"
    if approved is False:
        return "rejected", "respins" if ai_candidate else "nu e incident (confirmat)"
    if state.get("next"):
        return "awaiting", "așteaptă decizia"
    return "not_candidate", "nu e candidat"


# ---------------------------------------------------------------------------
# Stil vizual (Console CSS)
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
    .mia-card {
        background-color: var(--mia-surface);
        border: 1px solid var(--mia-border);
        border-radius: 3px;
        padding: 1rem 1.25rem;          /* acelasi spatiu sus si jos fata de text */
        margin-bottom: 0.75rem;
    }
    .mia-card .mia-cluster-title { margin: 0 0 0.4rem 0; line-height: 1.4; }
    .mia-card .mia-meta { margin: 0; line-height: 1.4; }

    /* chip-urile din filtre (multiselect) in culoarea aplicatiei, chiar daca tema nu e preluata */
    span[data-baseweb="tag"] { background-color: var(--mia-accent) !important; color: #FFFFFF !important; }

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
    st.markdown(
        f'<div class="mia-node-box">'
        f'<div class="node-label">Node: {node_name}</div>'
        f'{description}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _fmt_ts(value: str | None) -> str:
    """ISO string (de la server) -> 'YYYY-MM-DD HH:MM:SS'."""
    if not value:
        return "N/A"
    return str(value)[:19].replace("T", " ")


OUTCOME_LABELS = {
    "declared": "Declarat",
    "declared_override": "Declarat (nepropus de AI)",
    "rejected": "Respins",
    "confirmed_not_incident": "Nu e incident (confirmat)",
    "dismissed": "Nepropus de AI (fără review)",  # doar inregistrari mai vechi
}


def _outcome_key(d: dict) -> str:
    """Categoria afisata pentru o decizie salvata: combina decizia omului cu propunerea AI."""
    if d["outcome"] == "dismissed":
        return "dismissed"
    ai_candidate = (d.get("assessment") or {}).get("is_major_incident_candidate", True)
    if d["outcome"] == "declared":
        return "declared" if ai_candidate else "declared_override"
    return "rejected" if ai_candidate else "confirmed_not_incident"


def _decision_label(ai_candidate: bool, approved: bool) -> str:
    """Eticheta deciziei umane pentru pasul de aprobare si sumar."""
    if ai_candidate:
        return "Declared" if approved else "Rejected"
    return "Declarat (nepropus de AI)" if approved else "Nu e incident (confirmat)"


def _fmt_clock(value: str) -> str:
    """ISO string -> 'HH:MM:SS'."""
    return _fmt_ts(value)[11:19]


@st.cache_data(show_spinner=False)
def cached_embedding(text: str) -> list[float]:
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


STEPS = [
    "Tichete",
    "Detectie & clustering",
    "Evaluare AI (LLM + RAG)",
    "Aprobare humana",
    "Comunicari",
    "Sumar",
]

def _init_state() -> None:
    defaults = {
        "wizard_step": 0,
        "scenario_key": None,
        "stream_tickets": [],
        "stream_done": False,
        "detection_result": None,
        "selected_cluster_id": None,
        "thread_id": None,
        "cluster_threads": {},  # cluster_id -> thread_id (un thread per cluster evaluat)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_flow() -> None:
    for key in (
        "scenario_key", "stream_tickets", "stream_done", "detection_result",
        "selected_cluster_id"
    ):
        st.session_state[key] = [] if key == "stream_tickets" else (
            False if key == "stream_done" else None
        )
    st.session_state["thread_id"] = None  # thread nou se creeaza la urmatoarea pornire a grafului
    st.session_state["cluster_threads"] = {}
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


def _ticket_details(issue: dict, key_prefix: str) -> None:
    """Detalii complete ale unui tichet Jira (metadate + descriere)."""
    f = issue.get("fields") or {}
    components = ", ".join(c.get("name", "") for c in f.get("components") or []) or "N/A"
    meta = [
        ("ID", issue.get("key", "N/A")),
        ("Creat (UTC)", _fmt_ts(f.get("created"))),
        ("Serviciu", components),
        ("Prioritate", (f.get("priority") or {}).get("name", "N/A")),
        ("Status", (f.get("status") or {}).get("name", "N/A")),
        ("Raportat de", (f.get("reporter") or {}).get("name", "N/A")),
        ("Locație", f.get("location") or "N/A"),
        ("Etichete", ", ".join(f.get("labels") or []) or "N/A"),
    ]
    cols = st.columns(4)
    for i, (label, value) in enumerate(meta):
        cols[i % 4].caption(label)
        cols[i % 4].text(value)
    st.text_area(
        "Descriere",
        value=f.get("description") or "(fără descriere)",
        height=140,
        disabled=True,
        key=f"{key_prefix}_desc_{issue.get('key')}",
    )


def _save_review(ticket_key: str, created: str, widget_key: str) -> None:
    """Callback: salveaza starea aleasa de operator pentru un tichet fara cluster."""
    try:
        ticket_reviews.set_review(ticket_key, created, st.session_state[widget_key], DECIDED_BY)
    except Exception as exc:  # noqa: BLE001
        st.session_state["review_error"] = str(exc)


def _render_unclustered(detection: dict, tickets_by_key: dict) -> None:
    """Tichetele care nu au intrat in niciun cluster, cu marcarea starii de catre operator."""
    items = detection.get("unclustered") or []
    st.divider()
    st.subheader(f"Tichete fără cluster ({len(items)})")
    if not items:
        st.caption("Toate tichetele au intrat într-un cluster.")
        return

    try:
        reviews = ticket_reviews.list_reviews()
    except Exception as exc:  # noqa: BLE001
        reviews = {}
        st.warning(f"Nu pot citi stările tichetelor din audit.db: {exc}")
    if st.session_state.pop("review_error", None):
        st.warning("Nu am putut salva starea tichetului.")

    statuses = [reviews.get((i["key"], i["created"]), {}).get("status", "unreviewed") for i in items]
    st.caption(
        f"{statuses.count('unreviewed')} nerevizuite · {statuses.count('handled')} tratate individual · "
        f"{statuses.count('watch')} de urmărit. Sortate după apropierea de cel mai apropiat cluster "
        "(informativ, nu modifică clusterele)."
    )

    markers = {"unreviewed": "○", "handled": "✓", "watch": "◔"}
    status_codes = list(ticket_reviews.STATUS_LABELS)
    for item, status in zip(items, statuses):
        review = reviews.get((item["key"], item["created"]))
        sim = item["nearest_similarity"]
        sim_txt = f" · similar {sim:.2f} cu {item['nearest_cluster_id']}" if sim is not None else ""
        label = f"{markers[status]} {item['key']} · {item['service']} · {item['summary'][:70]}{sim_txt}"
        with st.expander(label):
            issue = tickets_by_key.get(item["key"])
            if issue:
                _ticket_details(issue, key_prefix="uncl")
            widget_key = f"review_{item['key']}_{item['created']}"
            st.radio(
                "Stare",
                options=status_codes,
                index=status_codes.index(status),
                format_func=lambda code: ticket_reviews.STATUS_LABELS[code],
                horizontal=True,
                key=widget_key,
                on_change=_save_review,
                args=(item["key"], item["created"], widget_key),
            )
            if review:
                st.caption(f"Marcat de {review['reviewed_by']} la {_fmt_ts(review['reviewed_at'])} UTC")


ACTION_LABELS = {
    "propose_major_incident": "Propus de AI",
    "assess_not_candidate": "Nepropus de AI",
    "approve_major_incident": "Incident aprobat",
    "reject_major_incident": "Incident respins",
    "confirm_not_incident": "Confirmat: nu e incident",
    "declare_major_incident": "Declarat de operator (nepropus de AI)",
    "draft_communication": "Comunicare generată",
    "edit_communication": "Comunicare editată",
    "approve_communication": "Comunicare aprobată",
    "reject_communication": "Comunicare neaprobată",
    "save_decision": "Decizie salvată",
    "remember_decision": "Salvat în memorie",
    "remember_decision_failed": "Eroare la salvarea în memorie",
}
AUDIENCE_LABELS = {"end_users": "End users", "management": "Management"}


def _render_incident_detail(d: dict) -> None:
    """Analiza detaliata a unui incident din istoric (decizie, evaluare AI, comunicari, cronologie, tichete)."""
    key = _outcome_key(d)
    css = "declared" if key in ("declared", "declared_override") else "rejected"

    st.divider()
    st.subheader(d["incident_id"])
    st.markdown(
        f'<div class="mia-meta">'
        f'<span class="mia-status {css}">{OUTCOME_LABELS[key]}</span>'
        f'&nbsp;&nbsp;{_severity_badge(d["severity"] or "N/A")}'
        f'&nbsp;&nbsp;{escape(str(d["service"] or "N/A"))} &nbsp;|&nbsp; {d["ticket_count"] or 0} tichete'
        f' &nbsp;|&nbsp; decis de <strong>{escape(str(d["decided_by"] or "N/A"))}</strong>'
        f' la {_fmt_ts(d["decided_at"])} UTC'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_decision, tab_ai, tab_comms, tab_timeline, tab_tickets = st.tabs(
        ["Decizie", "Evaluare AI", "Comunicări", "Cronologie (audit)", "Tichete"]
    )

    with tab_decision:
        st.markdown(f"**Cluster:** `{d['cluster_id']}`")
        verdicts = {
            "declared": "propus de AI și declarat de operator ca incident major.",
            "declared_override": "AI nu l-a propus, dar operatorul l-a declarat incident major.",
            "rejected": "propus de AI, dar respins de operator.",
            "confirmed_not_incident": "AI nu l-a propus, iar operatorul a confirmat că nu e incident major.",
            "dismissed": "AI nu l-a propus; nu a existat decizie umană (înregistrare mai veche).",
        }
        st.markdown(f"**Rezultat:** {verdicts[key]}")
        if d.get("reason"):
            label = "Raționamentul AI (început)" if key == "dismissed" else "Motiv"
            st.markdown(f"**{label}:**")
            st.write(d["reason"])

    with tab_ai:
        a = d.get("assessment")
        if not a:
            st.info("Nu există o evaluare AI salvată.")
        else:
            st.markdown(
                f'<div class="mia-meta">{_severity_badge(a["estimated_severity"])}'
                f'&nbsp;&nbsp;Candidat Major Incident: <strong>{"da" if a["is_major_incident_candidate"] else "nu"}</strong>'
                f'&nbsp;&nbsp;Confidence: <strong>{a["confidence"]:.2f}</strong>'
                f'&nbsp;&nbsp;Acțiune recomandată: <span class="mia-mono">{a["recommended_action"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Raționament**")
            st.write(a["reasoning"])
            sources = a.get("rag_sources") or []
            st.markdown("**Surse RAG**")
            st.caption(", ".join(sources) if sources else "(fără surse)")
            if any(s.startswith("MEM-") for s in sources):
                st.caption("Evaluarea a folosit și decizii umane anterioare (surse MEM-…).")

    with tab_comms:
        comms = d.get("communications")
        if not comms:
            st.info("Nu au fost generate comunicări (incidentul nu a fost declarat).")
        else:
            for audience, c in comms.items():
                with st.container(border=True):
                    st.markdown(
                        f'<div class="mia-comm-label">{AUDIENCE_LABELS.get(audience, audience)}</div>',
                        unsafe_allow_html=True,
                    )
                    status = "aprobată" if c.get("approved") else "neaprobată"
                    note = "editată de operator" if c.get("edited") else "text generat, needitat"
                    st.caption(f"{status} · {note}")
                    st.markdown(f"**{c['subject']}**")
                    st.write(c["body"])
                    st.caption(f"Surse: {', '.join(c.get('rag_sources') or []) or '(fără surse)'}")
                    if c.get("edited"):
                        with st.expander("Text original generat de AI"):
                            st.markdown(f"**{c.get('original_subject', '')}**")
                            st.write(c.get("original_body", ""))

    with tab_timeline:
        try:
            events = list(reversed(audit_store.list_audit_events(limit=500, incident_id=d["incident_id"])))
        except Exception as exc:  # noqa: BLE001
            events = []
            st.warning(f"Nu pot citi cronologia: {exc}")
        if not events:
            st.info("Nu există evenimente de audit pentru acest incident.")
        else:
            st.dataframe(
                [
                    {
                        "Ora (UTC)": _fmt_ts(e["timestamp"]),
                        "Cine": e["actor"],
                        "Acțiune": ACTION_LABELS.get(e["action"], e["action"]),
                        "Detalii": e["output_ref"] or e["input_ref"] or "",
                        "Model": e["model"] or "",
                        "Surse": ", ".join(e["rag_sources"]),
                    }
                    for e in events
                ],
                hide_index=True,
                use_container_width=True,
            )

    with tab_tickets:
        ticket_ids = d.get("ticket_ids") or []
        summaries = d.get("summaries") or []
        st.markdown("**Rezumatele trimise la evaluare**")
        for s in summaries:
            st.markdown(f"- {s}")
        st.markdown(f"**Tichete din cluster ({len(ticket_ids)})**")
        st.caption(
            "Detaliile complete sunt disponibile doar cât timp baza locală de tichete nu a fost "
            "resetată (butonul „Începe un incident nou”)."
        )
        for tkey in ticket_ids:
            issue = ticket_store.get_raw_issue(tkey)
            if issue:
                title = issue["fields"].get("summary", "")
                with st.expander(f"{tkey} · {title[:90]}"):
                    _ticket_details(issue, key_prefix=f"hist_{d['incident_id']}")
            else:
                st.caption(f"{tkey} (detalii indisponibile)")


def _render_history_page() -> None:
    """Pagina 'Istoric incidente': toate clusterele evaluate, cu filtre si analiza detaliata la selectie."""
    st.title("Istoric incidente")
    st.caption(
        "Toate clusterele evaluate: decizii umane și clustere nepropuse de AI. "
        "Selectează un rând din tabel pentru analiza detaliată."
    )

    try:
        decisions = audit_store.list_decisions(limit=500)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nu pot citi istoricul din audit.db: {exc}")
        return
    if not decisions:
        st.info("Nicio decizie înregistrată încă.")
        return

    counts = Counter(_outcome_key(d) for d in decisions)
    declared = counts.get("declared", 0) + counts.get("declared_override", 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(decisions))
    m2.metric("Declarate", declared)
    m3.metric("Nu sunt incidente", len(decisions) - declared)
    m4.metric("Operator ≠ propunerea AI", counts.get("declared_override", 0) + counts.get("rejected", 0))

    f1, f2, f3 = st.columns(3)
    outcomes = f1.multiselect(
        "Rezultat", options=list(OUTCOME_LABELS), default=list(OUTCOME_LABELS), format_func=OUTCOME_LABELS.get
    )
    services = f2.multiselect(
        "Serviciu", options=sorted({d["service"] for d in decisions if d["service"]}), placeholder="Toate serviciile"
    )
    severities = f3.multiselect(
        "Severitate", options=sorted({d["severity"] for d in decisions if d["severity"]}), placeholder="Toate"
    )

    filtered = [
        d for d in decisions
        if _outcome_key(d) in outcomes
        and (not services or d["service"] in services)
        and (not severities or d["severity"] in severities)
    ]
    if not filtered:
        st.info("Niciun rezultat pentru filtrele alese.")
        return

    # cheia depinde de filtre: selectia se reseteaza cand se schimba lista afisata
    table_key = "history_table_" + "|".join([",".join(sorted(outcomes)), ",".join(services), ",".join(severities)])
    event = st.dataframe(
        [
            {
                "Incident": d["incident_id"],
                "Rezultat": OUTCOME_LABELS[_outcome_key(d)],
                "Severitate": d["severity"],
                "Serviciu": d["service"],
                "Tichete": d["ticket_count"],
                "Decis de": d["decided_by"],
                "Data/ora (UTC)": _fmt_ts(d["decided_at"]),
                "Motiv": d["reason"] or "",
            }
            for d in filtered
        ],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    selected = event.selection.rows
    if selected and selected[0] < len(filtered):
        _render_incident_detail(filtered[selected[0]])
    else:
        st.caption("Selectează un incident din tabel pentru a-i vedea detaliile.")


@st.fragment(run_every=2)
def _live_tickets_view() -> None:
    """Tabelul cu tichete, reimprospatat automat la 2s."""
    rows = ticket_store.list_tickets(limit=200)
    st.caption(f"{ticket_store.count_tickets()} tichete în baza locală (actualizare automată la 2s, cele mai noi primele)")
    if rows:
        st.dataframe(
            [
                {
                    "ID": r["key"],
                    "Ora (UTC)": r["created"][11:19],
                    "Serviciu": r["service"],
                    "Prioritate": r["priority"],
                    "Rezumat": r["summary"],
                }
                for r in rows
            ],
            hide_index=True,
            use_container_width=True,
        )
        summaries = {r["key"]: r["summary"] for r in rows}
        chosen_key = st.selectbox(
            "Detalii tichet",
            options=list(summaries),
            index=None,
            placeholder="Alege un tichet pentru a vedea descrierea completă...",
            format_func=lambda k: f"{k} · {summaries[k][:80]}",
            key="live_ticket_detail",
        )
        if chosen_key:
            issue = ticket_store.get_raw_issue(chosen_key)
            if issue:
                with st.container(border=True):
                    _ticket_details(issue, key_prefix="live")
    else:
        st.info("Niciun tichet încă. Verifică dacă ingestorul rulează (python -m app.ingestion.ingestor).")


def _mock_post(path: str) -> dict | None:
    try:
        return httpx.post(f"{MOCK_URL}{path}", timeout=5).json()
    except httpx.HTTPError:
        return None


# Pornire automata a fluxului de tichete la intrarea in aplicatie (idempotent pe server)
if not st.session_state.get("sim_started"):
    if _mock_post("/mock/start") is not None:
        st.session_state["sim_started"] = True


if VIEW == VIEW_HISTORY:
    _render_history_page()
    st.stop()

st.title("Major Incident Agent")
st.caption(
    f"Flux agentic orchestrat de LangGraph Server ({LANGGRAPH_URL}). "
    "Urmareste executia live in LangGraph Studio."
)
if st.session_state["thread_id"]:
    st.caption(f"Thread curent: `{st.session_state['thread_id']}`")

_render_stepper_header(st.session_state["wizard_step"])

step = st.session_state["wizard_step"]

# ===========================================================================
# PASUL 1 - Sosire tichete (live)
# ===========================================================================
if step == 0:
    st.subheader(STEPS[0])
    _node_box(
        "Ticket Intake (live)",
        "Tichetele sosesc în timp real prin API-ul Jira, sunt preluate de ingestor și salvate "
        "într-un tabel local. Fluxul pornește automat la intrarea în aplicație; tu nu selectezi nimic.",
    )

    _live_tickets_view()

    # Butonul NU poate fi dezactivat pe baza numarului de tichete: tabelul se reimprospateaza
    # intr-un fragment, iar restul paginii nu se reexecuta, deci starea "disabled" ar ramane veche.
    if st.button("Next: Detecție & clustering"):
        snapshot = ticket_store.list_raw_issues()
        if not snapshot:
            st.warning("Încă nu a sosit niciun tichet.")
        else:
            # Snapshot cronologic, in formatul Jira folosit mai departe
            st.session_state["stream_tickets"] = snapshot
            st.session_state["stream_done"] = True
            st.session_state["detection_result"] = None
            st.session_state["cluster_threads"] = {}
            st.session_state["wizard_step"] = 1
            st.rerun()

# ===========================================================================
# PASUL 2 - Detecție & clustering
# ===========================================================================
elif step == 1:
    st.subheader(STEPS[1])
    _node_box(
        "Detection Pipeline",
        "Calculează embeddings BGE-M3 pe textul complet al tichetului (rezumat, descriere, serviciu, etichete) "
        "și similaritatea cosinus, apoi grupează tichetele corelate pe o fereastră glisantă de "
        f"{settings.clustering_window_minutes} min, unind clusterele suprapuse "
        f"(prag {settings.similarity_threshold}, minim {settings.min_tickets_per_cluster} tichete/cluster).",
    )

    tickets = st.session_state["stream_tickets"]

    if st.session_state["detection_result"] is None:
        if st.button("Rulează detecția"):
            with st.spinner("Se calculează embeddings + similaritate..."):
                raw_clusters, similarity_matrix = detect_clusters_with_matrix(tickets, embed=cached_embedding)
                clusters = [
                    build_incident_cluster(tickets, similarity_matrix, idx, seq)
                    for seq, idx in enumerate(raw_clusters, start=1)
                ]
                st.session_state["detection_result"] = {
                    "tickets": tickets,
                    "clusters": clusters,
                    "indices_by_cluster": {c.cluster_id: idx for c, idx in zip(clusters, raw_clusters)},
                    "unclustered": find_unclustered(
                        tickets, raw_clusters, similarity_matrix, [c.cluster_id for c in clusters]
                    ),
                }
                if len(clusters) == 1:
                    st.session_state["selected_cluster_id"] = clusters[0].cluster_id
            st.rerun()
    else:
        clusters = st.session_state["detection_result"]["clusters"]

        if not clusters:
            st.warning("Niciun cluster peste prag în acest scenariu.")
            if st.button("Back"):
                st.session_state["wizard_step"] = 0
                st.rerun()
        else:
            threads = st.session_state["cluster_threads"]
            by_id = {c.cluster_id: c for c in clusters}
            ids = list(by_id)
            statuses = {cid: _cluster_status(threads.get(cid)) for cid in ids}
            done = sum(1 for code, _ in statuses.values() if code in DONE_CODES)
            st.caption(f"{done} din {len(ids)} clustere finalizate")

            current = st.session_state["selected_cluster_id"]
            if current not in by_id:
                # primul cluster neevaluat, altfel primul din lista
                current = next((cid for cid in ids if statuses[cid][0] == "new"), ids[0])
            chosen = st.radio(
                "Clustere detectate",
                options=ids,
                index=ids.index(current),
                format_func=lambda cid: (
                    f"{cid} · {by_id[cid].service_guess} · {by_id[cid].ticket_count} tichete · {statuses[cid][1]}"
                ),
            )
            st.session_state["selected_cluster_id"] = chosen
            selected = by_id[chosen]
            code, label = statuses[chosen]

            st.markdown(
                f'<div class="mia-card">'
                f'<div class="mia-cluster-title">'
                f'<span class="id mia-mono">{escape(selected.cluster_id)}</span>'
                f'<span class="service">{escape(selected.service_guess)}</span>'
                f'</div>'
                f'<div class="mia-meta">'
                f'{selected.ticket_count} tichete &nbsp;|&nbsp; '
                f'similaritate {selected.centroid_similarity:.2f} &nbsp;|&nbsp; '
                f'{selected.window_start:%H:%M:%S}\u2013{selected.window_end:%H:%M:%S} '
                f'&nbsp;|&nbsp; stare: <strong>{escape(label)}</strong>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            tickets_by_key = {t["key"]: t for t in st.session_state["detection_result"]["tickets"]}
            with st.expander(f"Tichetele clusterului ({selected.ticket_count})"):
                for tkey in selected.ticket_ids:
                    issue = tickets_by_key.get(tkey)
                    if not issue:
                        st.caption(f"{tkey} (indisponibil)")
                        continue
                    title = issue["fields"].get("summary", "")
                    with st.expander(f"{tkey} · {title[:90]}"):
                        _ticket_details(issue, key_prefix=f"cluster_{selected.cluster_id}")

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Back"):
                    st.session_state["wizard_step"] = 0
                    st.rerun()
            with col_b:
                if code == "new":
                    if st.button("Next: Evaluare AI (Pornire Graf LangGraph)"):
                        detection = st.session_state["detection_result"]
                        indices = detection["indices_by_cluster"][selected.cluster_id]
                        summaries = [detection["tickets"][i]["fields"]["summary"] for i in indices]

                        with st.spinner("Se creează thread-ul și rulează node_assess_incident pe server..."):
                            thread = client.threads.create()
                            st.session_state["thread_id"] = thread["thread_id"]
                            # inregistrat inainte de rulare: clusterul apare "in evaluare" chiar daca runul esueaza
                            st.session_state["cluster_threads"][selected.cluster_id] = thread["thread_id"]
                            st.session_state["decision_reason"] = ""
                            st.session_state["declare_severity"] = "SEV2"
                            _run_graph(
                                input_payload={
                                    "cluster": selected.model_dump(mode="json"),
                                    "summaries": summaries,
                                }
                            )
                        st.session_state["wizard_step"] = 2
                        st.rerun()
                else:
                    button_label = "Vezi rezultatul" if code in DONE_CODES else "Continuă evaluarea"
                    if st.button(button_label):
                        st.session_state["thread_id"] = threads[selected.cluster_id]
                        st.session_state["wizard_step"] = OPEN_STEP[code]
                        st.rerun()

        _render_unclustered(
            st.session_state["detection_result"],
            {t["key"]: t for t in st.session_state["detection_result"]["tickets"]},
        )

# ===========================================================================
# PASUL 3 - Evaluare AI (LLM + RAG) - Executat în LangGraph Server
# ===========================================================================
elif step == 2:
    st.subheader(STEPS[2])
    _node_box(
        "Assessment Agent (node_assess_incident)",
        "Nodul din LangGraph interoghează RAG/ChromaDB și LLM Ollama pentru clasificare severitate "
        "și candidat Incident Major.",
    )

    thread_state = _get_thread_state()
    assessment = thread_state["values"].get("assessment")

    if assessment:
        st.markdown(
            f'<div class="mia-meta">'
            f'{_severity_badge(assessment["estimated_severity"])}'
            f'&nbsp;&nbsp;Candidat Major Incident: <strong>'
            f'{"da" if assessment["is_major_incident_candidate"] else "nu"}</strong>'
            f'&nbsp;&nbsp;Confidence: <strong>{assessment["confidence"]:.2f}</strong>'
            f'&nbsp;&nbsp;Acțiune recomandată: '
            f'<span class="mia-mono">{assessment["recommended_action"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Raționament și surse RAG"):
            st.write(assessment["reasoning"])
            st.caption(f"Surse: {', '.join(assessment['rag_sources'])}")

        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("Back"):
                st.session_state["wizard_step"] = 1
                st.rerun()
        with col_b:
            if st.button("Next: Aprobare umană"):
                st.session_state["wizard_step"] = 3
                st.rerun()
    else:
        st.info("Se procesează de către LangGraph...")
        if st.button("Reîncearcă pasul"):
            st.rerun()

# ===========================================================================
# PASUL 4 - Aprobare umană (HITL #1 via LangGraph interrupt)
# ===========================================================================
elif step == 3:
    st.subheader(STEPS[3])
    _node_box(
        "Human-in-the-loop #1 (node_human_review_incident)",
        "Graful LangGraph s-a întrerupt nativ (interrupt) și așteaptă o decizie umană "
        "transmisă prin resume.",
    )

    thread_state = _get_thread_state()
    values = thread_state["values"]
    pending_nodes = thread_state.get("next") or []
    assessment = values.get("assessment")
    user_approved = values.get("user_approved_incident")

    if assessment:
        st.markdown(
            f'<div class="mia-meta">Recomandare AI: {_severity_badge(assessment["estimated_severity"])} '
            f'&nbsp;&nbsp;acțiune: <span class="mia-mono">{assessment["recommended_action"]}</span></div>',
            unsafe_allow_html=True,
        )

    if user_approved is None and not pending_nodes:
        # Graful s-a incheiat fara sa ajunga la HITL (assessment: nu e candidat)
        st.info(
            "Evaluarea nu a propus Major Incident - graful s-a încheiat fără aprobare umană. "
            "Clusterul a fost salvat în istoric ca „Nepropus de AI”."
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("Back"):
                st.session_state["wizard_step"] = 2
                st.rerun()
        with col_b:
            if st.button("Next: Sumar"):
                st.session_state["wizard_step"] = 5
                st.rerun()

    elif user_approved is None:
        ai_candidate = (assessment or {}).get("is_major_incident_candidate", True)
        if ai_candidate:
            reason = st.text_input(
                "Motiv (recomandat la respingere)",
                key="decision_reason",
                placeholder="ex.: un singur utilizator afectat, nu e incident major",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Aprobă Major Incident"):
                    with st.spinner("Se transmite decizia către LangGraph (resume)..."):
                        _run_graph(resume={"approved": True, "decided_by": DECIDED_BY})
                    st.rerun()
            with col_b:
                if st.button("Respinge"):
                    with st.spinner("Se transmite respingerea către LangGraph (resume)..."):
                        _run_graph(resume={
                            "approved": False,
                            "reason": reason.strip(),
                            "decided_by": DECIDED_BY,
                        })
                    st.rerun()
        else:
            st.info(
                "Assessment nu a propus incident major. Decizia îți aparține: confirmă că nu e incident "
                "sau declară incident major."
            )
            reason = st.text_input(
                "Motiv (recomandat dacă declari incidentul)",
                key="decision_reason",
                placeholder="ex.: impact confirmat de echipa locală",
            )
            severity = st.selectbox("Severitate (dacă declari incidentul)", ["SEV2", "SEV1"], key="declare_severity")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Confirm: nu e incident"):
                    with st.spinner("Se transmite decizia către LangGraph (resume)..."):
                        _run_graph(resume={
                            "approved": False,
                            "reason": reason.strip(),
                            "decided_by": DECIDED_BY,
                        })
                    st.rerun()
            with col_b:
                if st.button("Declar incident major"):
                    with st.spinner("Se transmite decizia către LangGraph (resume)..."):
                        _run_graph(resume={
                            "approved": True,
                            "severity": severity,
                            "reason": reason.strip(),
                            "decided_by": DECIDED_BY,
                        })
                    st.rerun()
    else:
        ai_candidate = (assessment or {}).get("is_major_incident_candidate", True)
        status_label = _decision_label(ai_candidate, user_approved)
        status_class = "declared" if user_approved else "rejected"

        st.markdown(
            f'<div class="mia-meta">'
            f'<span class="mia-status {status_class}">{status_label}</span>'
            f'&nbsp;&nbsp;procesat și înregistrat în LangGraph State'
            f'</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("Back"):
                st.session_state["wizard_step"] = 2
                st.rerun()
        with col_b:
            if user_approved:
                if st.button("Next: Comunicări"):
                    st.session_state["wizard_step"] = 4
                    st.rerun()
            else:
                # nu s-a declarat incident: nu exista comunicari de aprobat, se merge direct la sumar
                if st.button("Next: Sumar"):
                    st.session_state["wizard_step"] = 5
                    st.rerun()

# ===========================================================================
# PASUL 5 - Comunicări (HITL #2 via LangGraph interrupt)
# ===========================================================================
elif step == 4:
    st.subheader(STEPS[4])

    thread_state = _get_thread_state()
    values = thread_state["values"]
    user_approved = values.get("user_approved_incident")

    if user_approved is not True:
        st.info("Nu există comunicări de aprobat - incidentul nu a fost declarat.")
        if st.button("Next: Sumar"):
            st.session_state["wizard_step"] = 5
            st.rerun()
    else:
        _node_box(
            "Communication Agent (node_generate_communications & HITL #2)",
            "LangGraph a generat comunicatele și a întrerupt execuția (node_human_review_communication) "
            "așteptând aprobarea finală pe audiențe.",
        )

        drafts = values.get("communication_drafts", {})
        comm_approvals = values.get("user_approved_communications", {})

        if drafts:
            thread_id = st.session_state["thread_id"]
            edited_audiences = values.get("communication_edits") or {}
            labels = {"end_users": "End users", "management": "Management"}

            if comm_approvals:
                # Dupa aprobare: text final, doar pentru citire
                for audience, draft in drafts.items():
                    with st.container(border=True):
                        st.markdown(f'<div class="mia-comm-label">{labels.get(audience, audience)}</div>', unsafe_allow_html=True)
                        st.markdown(f"**{draft['subject']}**")
                        st.write(draft["body"])
                        note = "editat de operator" if audience in edited_audiences else "text generat, needitat"
                        st.caption(f"Surse: {', '.join(draft['rag_sources'])} · {note}")
                st.markdown('<span class="mia-status approved">Comunicări Aprobate</span>', unsafe_allow_html=True)
            else:
                st.caption("Poți edita subiectul și textul înainte de aprobare. Versiunea finală (și originalul) se salvează.")
                edits: dict[str, dict] = {}
                approvals: dict[str, bool] = {}
                invalid = False
                for audience, draft in drafts.items():
                    with st.container(border=True):
                        st.markdown(f'<div class="mia-comm-label">{labels.get(audience, audience)}</div>', unsafe_allow_html=True)
                        subject = st.text_input(
                            "Subiect", value=draft["subject"], key=f"comm_subject_{thread_id}_{audience}"
                        )
                        body = st.text_area(
                            "Text", value=draft["body"], height=220, key=f"comm_body_{thread_id}_{audience}"
                        )
                        st.caption(f"Surse: {', '.join(draft['rag_sources'])}")
                        approvals[audience] = st.checkbox(
                            "Aprobă această comunicare", value=True, key=f"comm_ok_{thread_id}_{audience}"
                        )
                        if not subject.strip() or not body.strip():
                            invalid = True
                            st.error("Subiectul și textul nu pot fi goale.")
                        elif subject.strip() != draft["subject"] or body.strip() != draft["body"]:
                            edits[audience] = {"subject": subject.strip(), "body": body.strip()}
                            st.caption("Modificat față de textul generat.")

                if st.button("Aprobă și Finalizează Comunicările", disabled=invalid):
                    with st.spinner("Se confirmă aprobarea comunicatelor în LangGraph..."):
                        _run_graph(resume={
                            "approved_users": approvals.get("end_users", False),
                            "approved_mgmt": approvals.get("management", False),
                            "edits": edits,
                            "decided_by": DECIDED_BY,
                        })
                    st.rerun()

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Back"):
                    st.session_state["wizard_step"] = 3
                    st.rerun()
            with col_b:
                if st.button("Next: Sumar"):
                    st.session_state["wizard_step"] = 5
                    st.rerun()
        else:
            st.warning("Draft-urile de comunicare se generează...")
            if st.button("Reîncarcă"):
                st.rerun()

# ===========================================================================
# PASUL 6 - Sumar
# ===========================================================================
elif step == 5:
    st.subheader(STEPS[5])

    values = _get_thread_state()["values"]

    cluster = values.get("cluster")
    assessment = values.get("assessment")
    user_approved = values.get("user_approved_incident")
    comm_approvals = values.get("user_approved_communications", {})
    final_status = values.get("final_status")

    with st.container(border=True):
        if cluster:
            st.markdown(
                f"**Cluster:** `{cluster['cluster_id']}` \u2014 {cluster['service_guess']} "
                f"({cluster['ticket_count']} tichete)"
            )

        if assessment:
            st.markdown(
                f"**Evaluare AI:** {_severity_badge(assessment['estimated_severity'])} "
                f"&nbsp;acțiune recomandată: `{assessment['recommended_action']}`",
                unsafe_allow_html=True,
            )

        if user_approved is None:
            pending = (
                "nu a fost necesară (înregistrare mai veche, fără review)"
                if final_status == "DISMISSED_BY_ASSESSMENT"
                else "încă nu a fost luată"
            )
            st.markdown(f"**Decizie umană:** {pending}")
        else:
            ai_candidate = (assessment or {}).get("is_major_incident_candidate", True)
            status_label = _decision_label(ai_candidate, user_approved)
            status_class = "declared" if user_approved else "rejected"
            st.markdown(
                f'**Decizie umană:** <span class="mia-status {status_class}">{status_label}</span>',
                unsafe_allow_html=True,
            )

        if comm_approvals:
            approved_list = [k for k, v in comm_approvals.items() if v]
            st.markdown(f"**Comunicări aprobate:** {', '.join(approved_list) or 'niciuna'}")

        st.markdown(f"**LangGraph State Status:** `{final_status}`")

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("Înapoi la clustere"):
            # clusterele detectate si thread-urile lor se pastreaza; se alege urmatorul neevaluat
            st.session_state["selected_cluster_id"] = None
            st.session_state["wizard_step"] = 1
            st.rerun()
    with col_b:
        if st.button("Începe un incident nou"):
            # Curatare completa DOAR dupa finalizarea procesului: mock + tabel local, apoi flux nou.
            # audit.db (decizii + audit) NU se sterge.
            _mock_post("/mock/reset")
            ticket_store.reset_db()
            _mock_post("/mock/start")
            _reset_flow()
            st.rerun()