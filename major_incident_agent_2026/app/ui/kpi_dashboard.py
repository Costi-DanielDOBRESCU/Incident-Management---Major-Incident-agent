"""
app/ui/kpi_dashboard.py

Pagina "KPI Dashboard" (documentatie, sectiunea 7). Doar CITESTE din audit.db
(decizii, audit, arhiva tichete, legaturi tichete) si din ground_truth.json (date mock).
Nu modifica nimic in fluxul existent.

KPI-uri afisate:
  1. Time to Detect (TTD)            - primul tichet -> propunerea Assessment Agent
  2. Time to Communicate (MTTR sim.) - primul tichet -> notificare trimisa, vs baseline manual
  3. % interventie umana             - pasi umani / (pasi umani + pasi automatizati)
  4. Acuratete detectie              - Precision / Recall / F1 pe clustere, vs ground truth
  5. Duplicate evitate               - tichete din incidente reale legate la un incident declarat
  + Rata de aprobare a propunerilor AI
  + Incidente pe serviciu
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

import streamlit as st

GROUND_TRUTH_FILE = Path(__file__).resolve().parents[1] / "data" / "mock_tickets" / "ground_truth.json"

DEFAULT_BASELINE_MINUTES = 45.0  # estimare pentru identificare + comunicare manuala (configurabila in pagina)

OUTCOME_LABELS = {
    "declared": "Declarat (propus de AI)",
    "declared_override": "Declarat (nepropus de AI)",
    "rejected": "Respins de operator",
    "confirmed_not_incident": "Nu e incident (confirmat)",
    "dismissed": "Nepropus de AI (fără review)",
}

# Pasi umani (aprobari / decizii) vs pasi automatizati din flux (audit_events.action)
HUMAN_ACTIONS = {
    "approve_major_incident", "reject_major_incident", "declare_major_incident", "confirm_not_incident",
    "approve_communication", "reject_communication", "edit_communication",
}
AUTOMATED_ACTIONS = {
    "propose_major_incident", "assess_not_candidate", "draft_communication",
    "send_notification", "update_ticket_status", "link_tickets_to_parent",
}
ASSESS_ACTIONS = {"propose_major_incident", "assess_not_candidate"}


# ---------------------------------------------------------------------------
# Utilitare
# ---------------------------------------------------------------------------
def _parse_ts(value: Any) -> datetime | None:
    """ISO sau Jira ('2026-08-19T09:22:05.000+0000') -> datetime aware (UTC daca lipseste fusul)."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _minutes(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds() / 60
    return delta if delta >= 0 else None  # negativ = date inconsistente (ex. timestamp-uri mock)


def _outcome_key(d: dict) -> str:
    """Aceeasi logica ca in Istoric: combina decizia omului cu propunerea AI."""
    if d["outcome"] == "dismissed":
        return "dismissed"
    ai_candidate = (d.get("assessment") or {}).get("is_major_incident_candidate", True)
    if d["outcome"] == "declared":
        return "declared" if ai_candidate else "declared_override"
    return "rejected" if ai_candidate else "confirmed_not_incident"


def _ai_candidate(d: dict) -> bool:
    if d["outcome"] == "dismissed":
        return False
    return bool((d.get("assessment") or {}).get("is_major_incident_candidate", True))


def _fmt_min(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} min" if value < 120 else f"{value / 60:.1f} h"


# ---------------------------------------------------------------------------
# Calcul KPI (functii pure, testabile fara Streamlit / baze de date)
# ---------------------------------------------------------------------------
def compute_kpis(
    decisions: list[dict],
    events: list[dict],
    links: list[dict],
    first_created: dict[str, datetime],
    ground_truth: dict[str, dict],
    baseline_minutes: float = DEFAULT_BASELINE_MINUTES,
) -> dict[str, Any]:
    """
    decisions      - audit_store.list_decisions()
    events         - audit_store.list_audit_events()
    links          - tools.list_ticket_links()
    first_created  - {incident_id: momentul creat al primului tichet din cluster}
    ground_truth   - {ticket_id: eticheta ground truth} (gol daca nu exista fisierul)
    """
    # --- timestamp-uri din audit, pe incident ---
    assess_at: dict[str, datetime] = {}
    comm_at: dict[str, datetime] = {}
    for ev in sorted(events, key=lambda e: e["timestamp"]):  # cronologic
        inc, ts = ev.get("incident_id"), _parse_ts(ev["timestamp"])
        if not inc or ts is None:
            continue
        if ev["action"] in ASSESS_ACTIONS:
            assess_at.setdefault(inc, ts)
        elif ev["action"] == "send_notification":
            comm_at.setdefault(inc, ts)

    # --- rand per incident (TTD / Time to Communicate) ---
    rows = []
    for d in decisions:
        inc = d["incident_id"]
        start = first_created.get(inc)
        rows.append({
            "incident_id": inc,
            "service": d.get("service") or "N/A",
            "severity": d.get("severity") or "N/A",
            "outcome_key": _outcome_key(d),
            "decided_at": d.get("decided_at"),
            "ttd": _minutes(start, assess_at.get(inc)),
            "ttc": _minutes(start, comm_at.get(inc)),
        })
    ttd_vals = [r["ttd"] for r in rows if r["ttd"] is not None]
    ttc_vals = [r["ttc"] for r in rows if r["ttc"] is not None]
    ttc_avg = mean(ttc_vals) if ttc_vals else None

    # --- % interventie umana ---
    human = sum(1 for e in events if e["action"] in HUMAN_ACTIONS)
    auto = sum(1 for e in events if e["action"] in AUTOMATED_ACTIONS)
    human_pct = human / (human + auto) * 100 if (human + auto) else None

    # --- acuratete detectie (la nivel de cluster) ---
    tp = fp = fn = tn = 0
    evaluated = 0
    for d in decisions:
        labels = [ground_truth[t] for t in (d.get("ticket_ids") or []) if t in ground_truth]
        if not labels:
            continue
        evaluated += 1
        truth_major = sum(1 for l in labels if l.get("is_major_incident_ticket")) / len(labels) >= 0.5
        pred = _ai_candidate(d)
        tp += pred and truth_major
        fp += pred and not truth_major
        fn += (not pred) and truth_major
        tn += (not pred) and not truth_major
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if evaluated and (tp or fp or fn) else None)

    # --- duplicate evitate ---
    declared_ids = {d["incident_id"] for d in decisions if d["outcome"] == "declared"}
    linked_keys = {l["ticket_key"] for l in links if l.get("parent_incident_id") in declared_ids}
    real_cluster_tickets = {
        t for d in decisions for t in (d.get("ticket_ids") or [])
        if ground_truth.get(t, {}).get("is_major_incident_ticket")
    }
    if ground_truth:
        linked_ok = len(real_cluster_tickets & linked_keys)
        dup_rate = linked_ok / len(real_cluster_tickets) * 100 if real_cluster_tickets else None
    else:
        linked_ok, dup_rate = len(linked_keys), None

    # --- rata de aprobare (doar propuneri AI evaluate de om) ---
    proposed = [d for d in decisions if _ai_candidate(d) and d["outcome"] in ("declared", "rejected")]
    approved = sum(1 for d in proposed if d["outcome"] == "declared")
    approval_rate = approved / len(proposed) * 100 if proposed else None

    return {
        "rows": rows,
        "n_decisions": len(decisions),
        "n_declared": len(declared_ids),
        "ttd_avg": mean(ttd_vals) if ttd_vals else None,
        "ttd_n": len(ttd_vals),
        "ttc_avg": ttc_avg,
        "ttc_n": len(ttc_vals),
        "baseline": baseline_minutes,
        "ttc_gain_pct": (1 - ttc_avg / baseline_minutes) * 100 if ttc_avg is not None and baseline_minutes else None,
        "human_steps": human, "auto_steps": auto, "human_pct": human_pct,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "evaluated": evaluated,
        "precision": precision, "recall": recall, "f1": f1,
        "dup_linked": linked_ok, "dup_total": len(real_cluster_tickets), "dup_rate": dup_rate,
        "has_ground_truth": bool(ground_truth),
        "proposed": len(proposed), "approved": approved, "approval_rate": approval_rate,
        "outcomes": Counter(r["outcome_key"] for r in rows),
    }


# ---------------------------------------------------------------------------
# Incarcare date (doar citire)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _load_ground_truth() -> dict[str, dict]:
    try:
        data = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {g["ticket_id"]: g for g in data if "ticket_id" in g}


@st.cache_data(ttl=10, show_spinner=False)
def _load_data() -> dict[str, Any]:
    from app.execution import audit_store, ticket_archive
    from app.execution import tools as execution_tools

    decisions = audit_store.list_decisions(limit=500)
    events = audit_store.list_audit_events(limit=20000)
    links = execution_tools.list_ticket_links()

    first_created: dict[str, datetime] = {}
    for d in decisions:
        issues = ticket_archive.get_incident_tickets(d["incident_id"])
        created = [_parse_ts(i.get("fields", {}).get("created")) for i in issues.values()]
        created = [c for c in created if c is not None]
        if created:
            first_created[d["incident_id"]] = min(created)
    return {"decisions": decisions, "events": events, "links": links, "first_created": first_created}


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f}%"


def render_kpi_page() -> None:
    st.markdown(
        '<div class="itsm-header-banner"><div>'
        '<div class="itsm-header-title">KPI Dashboard</div>'
        '<div class="itsm-header-sub">Agentic (MIA) vs proces manual &bull; calculat din audit.db și ground truth (date mock)</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    top_l, top_r = st.columns([4, 1])
    with top_l:
        baseline = st.number_input(
            "Baseline manual: timp estimat până la comunicare (minute)",
            min_value=1.0, value=DEFAULT_BASELINE_MINUTES, step=5.0,
            help="Estimare pentru procesul AS-IS (identificare manuală + comunicare). Ajustează-o după nevoie.",
        )
    with top_r:
        st.write("")
        if st.button("Reîncarcă datele", use_container_width=True):
            _load_data.clear()
            st.rerun()

    try:
        data = _load_data()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nu pot citi datele din audit.db: {exc}")
        return
    if not data["decisions"]:
        st.info("Nicio decizie înregistrată încă. Rulează cel puțin un incident din „Flux incident”.")
        return

    k = compute_kpis(
        data["decisions"], data["events"], data["links"], data["first_created"],
        _load_ground_truth(), baseline_minutes=baseline,
    )

    # --- Rand 1: 5 KPI din documentatie ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Time to Detect", _fmt_min(k["ttd_avg"]), help=f"Primul tichet → propunerea Assessment. Calculat pe {k['ttd_n']} incidente.")
    c2.metric(
        "Time to Communicate", _fmt_min(k["ttc_avg"]),
        delta=None if k["ttc_gain_pct"] is None else f"{-k['ttc_gain_pct']:.0f}% vs manual ({_fmt_min(baseline)})",
        delta_color="inverse",
        help=f"MTTR simulat: primul tichet → notificare trimisă. Calculat pe {k['ttc_n']} incidente declarate.",
    )
    c3.metric("% intervenție umană", _pct(k["human_pct"]),
              help=f"{k['human_steps']} pași umani (aprobări/decizii) din {k['human_steps'] + k['auto_steps']} pași de flux.")
    c4.metric("F1 detecție", "N/A" if k["f1"] is None else f"{k['f1']:.2f}",
              help="Calculat la nivel de cluster, față de ground truth (doar date mock).")
    c5.metric("Duplicate evitate", _pct(k["dup_rate"]) if k["has_ground_truth"] else f"{k['dup_linked']} tichete",
              help=f"{k['dup_linked']} din {k['dup_total']} tichete ale incidentelor reale au fost legate la un incident declarat.")

    # --- Rand 2: aprobare + acuratete detaliata ---
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rata de aprobare propuneri AI", _pct(k["approval_rate"]),
              help=f"{k['approved']} aprobate din {k['proposed']} propuneri AI evaluate de operator.")
    d2.metric("Precision", "N/A" if k["precision"] is None else f"{k['precision']:.2f}")
    d3.metric("Recall", "N/A" if k["recall"] is None else f"{k['recall']:.2f}")
    d4.metric("Clustere evaluate", k["evaluated"], help=f"TP={k['tp']} · FP={k['fp']} · FN={k['fn']} · TN={k['tn']}")
    if not k["has_ground_truth"]:
        st.caption("ground_truth.json nu a fost găsit: KPI-urile de acuratețe și duplicate nu pot fi calculate.")
    else:
        st.caption("Precision/Recall se calculează doar pe clusterele detectate și evaluate; "
                   "incidentele reale nedetectate de clustering nu apar aici.")

    # --- Grafice ---
    import pandas as pd

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Decizii: AI vs operator**")
        df_out = pd.DataFrame(
            {"Număr": [k["outcomes"].get(key, 0) for key in OUTCOME_LABELS]},
            index=list(OUTCOME_LABELS.values()),
        )
        st.bar_chart(df_out, horizontal=True)
    with g2:
        st.markdown("**Incidente pe serviciu**")
        only_declared = st.toggle("Doar incidente declarate", value=True)
        sel = [r for r in k["rows"] if not only_declared or r["outcome_key"] in ("declared", "declared_override")]
        if sel:
            df = pd.DataFrame(sel).groupby(["service", "severity"]).size().unstack(fill_value=0)
            st.bar_chart(df)
        else:
            st.info("Niciun incident pentru selecția curentă.")

    # --- Tabel detaliu ---
    st.markdown("**Detaliu pe incident**")
    st.dataframe(
        [
            {
                "Incident": r["incident_id"],
                "Rezultat": OUTCOME_LABELS[r["outcome_key"]],
                "Serviciu": r["service"],
                "Severitate": r["severity"],
                "TTD": _fmt_min(r["ttd"]),
                "Time to Communicate": _fmt_min(r["ttc"]),
                "Decis la (UTC)": str(r["decided_at"] or "")[:19].replace("T", " "),
            }
            for r in k["rows"]
        ],
        hide_index=True,
        use_container_width=True,
    )