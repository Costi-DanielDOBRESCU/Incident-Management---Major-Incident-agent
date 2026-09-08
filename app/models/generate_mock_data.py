"""
Genereaza date mock pentru MIA:
  - app/data/mock_tickets/tickets.json        (format Jira-like, sectiunea 5.4)
  - app/data/mock_tickets/ground_truth.json   (etichete pentru evaluare KPI, sectiunea 7)
  - app/data/knowledge_base/historical_major_incidents.json
  - app/data/knowledge_base/runbooks.json
  - app/data/knowledge_base/communication_templates.json

Rulare:  python -m scripts.generate_mock_data
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.schemas import GroundTruthLabel, KnowledgeBaseDocument, Ticket  # noqa: E402

random.seed(42)  # reproductibil - important pt demo si pt teste

OUT_TICKETS = ROOT / "app" / "data" / "mock_tickets"
OUT_KB = ROOT / "app" / "data" / "knowledge_base"
OUT_TICKETS.mkdir(parents=True, exist_ok=True)
OUT_KB.mkdir(parents=True, exist_ok=True)

LOCATIONS = [
    "RO-Timisoara", "RO-Bucuresti", "RO-Cluj", "DE-Munich",
    "US-Austin", "US-NewYork", "UK-London", "PL-Warsaw",
]

# ============================================================
# 1. Cataloage de servicii + cauze reale + template-uri de parafrazare
#    Fiecare cauza are variante "tehnice" si "non-tehnice" de formulare,
#    ca sa testam robustetea embeddings-urilor (nu doar keyword match).
# ============================================================

SERVICE_CAUSES = {
    "VPN Gateway": {
        "category": "Network",
        "causes": {
            "vpn_auth_degradation": {
                "label": "VPN authentication service degradation",
                "templates": [
                    ("Cannot access corporate VPN",
                     "Unable to connect to VPN since {time}, error 'authentication timeout'."),
                    ("VPN connection failing",
                     "VPN client shows 'connection failed' repeatedly when trying to authenticate."),
                    ("Remote employees cannot connect to VPN",
                     "Several colleagues report they cannot log into the VPN this morning."),
                    ("VPN authentication unavailable",
                     "Getting 'authentication service unavailable' when starting the VPN tunnel."),
                    ("Unable to establish VPN connection",
                     "VPN handshake fails at the auth step, worked fine yesterday."),
                    ("VPN login stuck on authenticating",
                     "The VPN app just spins on 'Authenticating...' and then times out."),
                    ("Can't login to VPN from home",
                     "Working from home today and the VPN just won't let me in, keeps failing."),
                ],
            },
            "vpn_licensing": {
                "label": "VPN licensing issue (per-device)",
                "templates": [
                    ("Unable to activate VPN license on new laptop",
                     "Just got a new laptop, VPN client says license not activated for this device."),
                    ("VPN license expired notification",
                     "VPN client shows 'license expired' popup, cannot start a session."),
                    ("New device VPN activation error",
                     "Trying to set up VPN on my replacement machine, activation fails with license error."),
                ],
            },
        },
    },
    "Email/Exchange": {
        "category": "Application",
        "causes": {
            "exchange_db_corruption": {
                "label": "Exchange mailbox database corruption",
                "templates": [
                    ("Cannot open mailbox in Outlook",
                     "Outlook shows 'cannot open your default e-mail folders' since {time}."),
                    ("Emails not loading in Outlook",
                     "Outlook is stuck, mailbox won't sync, spinning wheel for several minutes."),
                    ("Mailbox inaccessible error",
                     "Getting a mailbox corruption error whenever I try to open my inbox."),
                    ("Outlook keeps crashing on startup",
                     "Outlook crashes right after login, worked fine until this morning."),
                    ("Unable to send or receive emails",
                     "No emails coming in or going out since around {time}, mailbox seems frozen."),
                    ("Exchange server not responding",
                     "Outlook shows 'trying to connect' constantly, never actually connects."),
                ],
            },
            "mail_queue_backlog": {
                "label": "Outbound mail queue backlog (spam filter)",
                "templates": [
                    ("Outgoing emails delayed by hours",
                     "Sent emails are taking hours to be delivered, recipients complain of delays."),
                    ("Emails stuck in outbox",
                     "My sent emails sit in the outbox for a long time before going out."),
                ],
            },
        },
    },
    "ERP System": {
        "category": "Application",
        "causes": {
            "erp_server_crash": {
                "label": "ERP application server crash (memory leak)",
                "templates": [
                    ("ERP application not responding",
                     "The ERP system just froze on me, page won't load since {time}."),
                    ("Cannot log into ERP system",
                     "Login page for the ERP portal times out, tried multiple times."),
                    ("ERP shows internal server error",
                     "Getting a 500 internal server error every time I open the ERP dashboard."),
                    ("ERP application crashed mid-transaction",
                     "Was entering a purchase order and the ERP app crashed, lost my data."),
                    ("ERP system extremely slow",
                     "Every screen in the ERP takes over a minute to load, basically unusable."),
                ],
            },
            "erp_db_pool_exhausted": {
                "label": "ERP database connection pool exhausted",
                "templates": [
                    ("ERP database connection error",
                     "ERP shows 'unable to connect to database' intermittently."),
                    ("Intermittent ERP timeouts",
                     "ERP works for a bit then times out, then works again, very inconsistent."),
                ],
            },
        },
    },
    "Network/Switch": {
        "category": "Infrastructure",
        "causes": {
            "core_switch_failure": {
                "label": "Core switch stack failure (data center)",
                "templates": [
                    ("No network connectivity in the office",
                     "Entire floor lost network access since {time}, wired and wifi both down."),
                    ("Internal network down",
                     "Cannot reach any internal systems, network seems completely down."),
                    ("Intermittent network drops",
                     "Network keeps dropping every few minutes across the whole building."),
                    ("Cannot access shared drives",
                     "Shared network drives are unreachable, 'network path not found' errors."),
                    ("Wifi and LAN both unstable",
                     "Both wired and wireless connections keep disconnecting randomly."),
                ],
            },
            "isp_circuit_flapping": {
                "label": "ISP circuit flapping (branch office)",
                "templates": [
                    ("Branch office internet keeps dropping",
                     "Internet connection at the branch office cuts out every 10-15 minutes."),
                    ("Intermittent internet outage at remote site",
                     "Remote site loses internet access sporadically throughout the day."),
                ],
            },
        },
    },
    "Cloud Storage": {
        "category": "Infrastructure",
        "causes": {
            "storage_sync_outage": {
                "label": "Cloud storage sync service outage",
                "templates": [
                    ("Files not syncing to cloud storage",
                     "Cloud storage client stuck on 'syncing' since {time}, no progress."),
                    ("Cannot upload files to cloud drive",
                     "Every upload attempt to the cloud drive fails with a generic error."),
                    ("Cloud storage shows sync error",
                     "Getting a persistent sync error icon on all files in the cloud folder."),
                    ("Shared cloud folders not updating",
                     "Colleagues' changes aren't showing up in the shared cloud folder."),
                ],
            },
            "storage_quota_exceeded": {
                "label": "Storage quota exceeded (tenant-level)",
                "templates": [
                    ("Cannot save new files, quota exceeded error",
                     "Getting a storage quota exceeded message when saving new documents."),
                    ("Cloud storage full notification",
                     "Received a notification that cloud storage is full, cannot add new files."),
                ],
            },
        },
    },
    "Internal Portal": {
        "category": "Application",
        "causes": {
            "portal_sso_failure": {
                "label": "Portal SSO integration failure",
                "templates": [
                    ("Cannot log into internal portal",
                     "Single sign-on redirect loop when trying to access the internal portal."),
                    ("Portal login redirect error",
                     "Portal keeps redirecting back to login page after entering credentials."),
                    ("SSO authentication failed on portal",
                     "Getting 'SSO authentication failed' every time I try to log into the portal."),
                    ("Internal portal inaccessible",
                     "Portal page won't load past the login screen since {time}."),
                ],
            },
            "portal_cdn_cache_bug": {
                "label": "Portal CDN cache invalidation bug",
                "templates": [
                    ("Portal showing outdated content",
                     "Internal portal is displaying old/stale content that should have updated."),
                    ("Portal pages not refreshing",
                     "Changes made to portal pages aren't reflecting for other users."),
                ],
            },
        },
    },
    "Print Services": {
        "category": "Infrastructure",
        "causes": {
            "print_server_crash": {
                "label": "Print server service crash",
                "templates": [
                    ("Cannot print to any office printer",
                     "None of the printers on the floor are responding since {time}."),
                    ("Print jobs stuck in queue",
                     "My print jobs just sit in the queue, nothing comes out of the printer."),
                    ("Printer offline error for everyone",
                     "All shared printers show 'offline' status, multiple people affected."),
                ],
            },
            "printer_driver_corruption": {
                "label": "Printer driver corruption after Windows update",
                "templates": [
                    ("Printer not working after Windows update",
                     "Since the latest Windows update, my printer no longer works, driver error."),
                    ("Print spooler error after update",
                     "Getting a print spooler service error since the recent update rolled out."),
                ],
            },
        },
    },
}

REPORTER_POOL = [f"user_{i:03d}" for i in range(100, 900)]


def rand_reporter() -> str:
    return random.choice(REPORTER_POOL)


def rand_location() -> str:
    return random.choice(LOCATIONS)


def fill_template(summary: str, description: str, ts: datetime) -> tuple[str, str]:
    time_str = ts.strftime("%H:%M")
    return summary, description.replace("{time}", time_str)


# ============================================================
# 2. Generare timeline: alternanta perioade calme / burst-uri
# ============================================================

def generate_tickets() -> tuple[list[Ticket], list[GroundTruthLabel]]:
    tickets: list[Ticket] = []
    labels: list[GroundTruthLabel] = []
    ticket_counter = 10000
    incident_group_counter = 1

    start = datetime(2026, 8, 1, 8, 0, tzinfo=timezone.utc)
    cursor = start

    all_services = list(SERVICE_CAUSES.keys())

    def next_ticket_id() -> str:
        nonlocal ticket_counter
        ticket_counter += 1
        return f"INC-{ticket_counter}"

    def make_ticket(service: str, cause_key: str, cause_info: dict, ts: datetime, priority: str) -> Ticket:
        summary_tpl, desc_tpl = random.choice(cause_info["templates"])
        summary, description = fill_template(summary_tpl, desc_tpl, ts)
        return Ticket(
            id=next_ticket_id(),
            source="jira_mock",
            created_at=ts,
            reporter=rand_reporter(),
            service=service,
            summary=summary,
            description=description,
            category=SERVICE_CAUSES[service]["category"],
            priority=priority,
            status="Open",
            location=rand_location(),
        )

    # --- Plan: ~10 burst-uri (Major Incident real) + cateva "false positive traps" + zgomot izolat ---
    N_BURSTS = 10
    N_FALSE_POSITIVE_TRAPS = 5

    burst_plan = []
    for _ in range(N_BURSTS):
        service = random.choice(all_services)
        cause_key = random.choice(list(SERVICE_CAUSES[service]["causes"].keys()))
        burst_plan.append((service, cause_key))

    total_events = N_BURSTS + N_FALSE_POSITIVE_TRAPS

    for event_i in range(total_events):
        # --- perioada calma: cateva tichete izolate, cauze random, neconectate ---
        n_quiet = random.randint(8, 20)
        for _ in range(n_quiet):
            cursor += timedelta(minutes=random.randint(15, 90))
            service = random.choice(all_services)
            cause_key = random.choice(list(SERVICE_CAUSES[service]["causes"].keys()))
            cause_info = SERVICE_CAUSES[service]["causes"][cause_key]
            priority = random.choice(["P2", "P3", "P3", "P4"])
            t = make_ticket(service, cause_key, cause_info, cursor, priority)
            tickets.append(t)
            labels.append(GroundTruthLabel(
                ticket_id=t.id, incident_group_id=None,
                is_major_incident_ticket=False, root_cause_id=cause_key,
                is_false_positive_trap=False,
            ))

        cursor += timedelta(minutes=random.randint(20, 60))

        if event_i < N_BURSTS:
            # --- BURST real: 5-15 tichete similare, fereastra 10-20 min ---
            service, cause_key = burst_plan[event_i]
            cause_info = SERVICE_CAUSES[service]["causes"][cause_key]
            n_burst = random.randint(5, 15)
            window_minutes = random.randint(10, 20)
            group_id = f"GT-{incident_group_counter:03d}"
            incident_group_counter += 1
            priority = random.choice(["P1", "P2"])

            burst_start = cursor
            for _ in range(n_burst):
                ts = burst_start + timedelta(minutes=random.uniform(0, window_minutes))
                t = make_ticket(service, cause_key, cause_info, ts, priority)
                tickets.append(t)
                labels.append(GroundTruthLabel(
                    ticket_id=t.id, incident_group_id=group_id,
                    is_major_incident_ticket=True, root_cause_id=cause_key,
                    is_false_positive_trap=False,
                ))
            cursor = burst_start + timedelta(minutes=window_minutes)
        else:
            # --- FALSE POSITIVE TRAP: aceeasi service, formulare similara, cauza DIFERITA ---
            service = random.choice(all_services)
            causes = list(SERVICE_CAUSES[service]["causes"].keys())
            if len(causes) < 2:
                continue
            trap_cause_key = causes[-1]
            cause_info = SERVICE_CAUSES[service]["causes"][trap_cause_key]
            n_trap = random.randint(2, 3)
            window_minutes = random.randint(10, 20)
            trap_start = cursor
            for _ in range(n_trap):
                ts = trap_start + timedelta(minutes=random.uniform(0, window_minutes))
                t = make_ticket(service, trap_cause_key, cause_info, ts, random.choice(["P2", "P3"]))
                tickets.append(t)
                labels.append(GroundTruthLabel(
                    ticket_id=t.id, incident_group_id=None,
                    is_major_incident_ticket=False, root_cause_id=trap_cause_key,
                    is_false_positive_trap=True,
                ))
            cursor = trap_start + timedelta(minutes=window_minutes)

    tickets.sort(key=lambda t: t.created_at)
    return tickets, labels


# ============================================================
# 3. Export in format Jira-like (sectiunea 5.4)
# ============================================================

def ticket_to_jira_issue(t: Ticket) -> dict:
    return {
        "key": t.id,
        "fields": {
            "summary": t.summary,
            "description": t.description,
            "created": t.created_at.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
            "priority": {"name": f"{t.priority} - {'High' if t.priority in ('P1', 'P2') else 'Medium'}"},
            "status": {"name": t.status},
            "components": [{"name": t.service}],
            "reporter": {"name": t.reporter},
            "labels": [t.category.lower()],
            "location": t.location,
        },
    }


# ============================================================
# 4. Knowledge Base: post-mortems, runbook-uri, template-uri comunicare
# ============================================================

def build_knowledge_base() -> list[KnowledgeBaseDocument]:
    docs: list[KnowledgeBaseDocument] = []

    post_mortems = [
        ("PM-2025-0117", "VPN Gateway", "VPN outage caused by expired auth certificate",
         "On 2025-11-14, the VPN authentication certificate expired unexpectedly, causing all "
         "authentication attempts to fail for approximately 3 hours. Root cause: certificate "
         "renewal automation had silently failed 30 days prior. Resolution: certificate manually "
         "renewed, monitoring alert added for certificate expiry (30/14/7 day warnings). "
         "45 tickets were raised in a 20-minute window before declaration.",
         ["vpn", "authentication", "certificate"]),
        ("PM-2025-0089", "Email/Exchange", "Exchange outage due to database corruption after failed patch",
         "A failed cumulative update left the Exchange mailbox database in an inconsistent state. "
         "Users were unable to access mailboxes for 90 minutes. Resolution: database repaired from "
         "backup, patch rollback procedure documented. 30+ correlated tickets in 15 minutes.",
         ["exchange", "email", "database"]),
        ("PM-2025-0154", "ERP System", "ERP outage from database connection pool exhaustion",
         "A runaway batch job exhausted the ERP database connection pool, causing intermittent "
         "failures for all users. Resolution: connection pool limits tuned, batch job rate-limited. "
         "Declared SEV2 after 12 correlated tickets in 25 minutes.",
         ["erp", "database", "connection-pool"]),
        ("PM-2025-0201", "Network/Switch", "Data center core switch stack failure",
         "A firmware bug caused the core switch stack to fail over repeatedly, causing intermittent "
         "connectivity loss across the building for 2 hours. Resolution: firmware rolled back, "
         "vendor engaged. Declared SEV1 due to full-building impact, 60+ tickets in 10 minutes.",
         ["network", "switch", "datacenter"]),
        ("PM-2025-0132", "Cloud Storage", "Cloud storage sync outage due to provider-side incident",
         "Upstream cloud storage provider had a regional outage affecting sync services. "
         "Declared SEV2, no internal root cause. Resolution: provider status page monitored, "
         "users notified to work locally until restored. 25 tickets in 18 minutes.",
         ["cloud-storage", "sync", "vendor-outage"]),
        ("PM-2025-0176", "Internal Portal", "Portal SSO outage due to identity provider misconfiguration",
         "A misapplied configuration change to the SSO identity provider caused a redirect loop "
         "for all portal users. Resolution: configuration reverted, change process tightened to "
         "require peer review. Declared SEV2, 20 tickets in 12 minutes.",
         ["portal", "sso", "identity"]),
        ("PM-2025-0143", "Print Services", "Print server outage from Windows Print Spooler vulnerability patch",
         "An emergency security patch for the Print Spooler service caused the print server to "
         "crash repeatedly. Resolution: patched service restarted with a stability fix, vendor "
         "hotfix applied. Declared SEV3 due to limited business impact, 18 tickets in 20 minutes.",
         ["print", "spooler", "windows-update"]),
    ]
    for doc_id, service, summary, content, tags in post_mortems:
        docs.append(KnowledgeBaseDocument(
            doc_id=doc_id, type="post_mortem", service=service,
            summary=summary, content=content, tags=tags,
        ))

    runbooks = [
        ("RB-VPN-002", "VPN Gateway",
         "Severity criteria for VPN Gateway incidents",
         "SEV1: Total VPN outage, all users affected, >30 min. SEV2: Partial outage or "
         "authentication degradation affecting multiple users (3+) within a 20-minute window. "
         "SEV3: Isolated single-user VPN issue (licensing, local config). "
         "Multi-user impact within a short window is the primary signal for SEV2+.",
         ["vpn", "severity", "runbook"]),
        ("RB-EXCH-001", "Email/Exchange",
         "Severity criteria for Exchange/Email incidents",
         "SEV1: Full mail service outage (send+receive) org-wide. SEV2: Mailbox access issues "
         "affecting 3+ users within 15 minutes, or delayed mail delivery org-wide. "
         "SEV3: Individual mailbox issues, spam filter false positives.",
         ["exchange", "severity", "runbook"]),
        ("RB-ERP-003", "ERP System",
         "Severity criteria for ERP incidents",
         "SEV1: ERP fully unavailable, blocking business transactions org-wide. SEV2: "
         "Degraded performance or intermittent errors affecting 3+ users within a short window. "
         "SEV3: Single-user or single-module issue.",
         ["erp", "severity", "runbook"]),
        ("RB-NET-004", "Network/Switch",
         "Severity criteria for Network/Switch incidents",
         "SEV1: Full site network outage. SEV2: Partial network degradation affecting a floor "
         "or department, 3+ correlated reports within 20 minutes. SEV3: Single connection issue.",
         ["network", "severity", "runbook"]),
        ("RB-CLOUD-005", "Cloud Storage",
         "Severity criteria for Cloud Storage incidents",
         "SEV1: Complete sync/upload outage org-wide. SEV2: Degraded sync affecting multiple "
         "users/teams within a short window. SEV3: Individual quota or permission issue.",
         ["cloud-storage", "severity", "runbook"]),
        ("RB-PORTAL-006", "Internal Portal",
         "Severity criteria for Internal Portal incidents",
         "SEV1: Portal fully inaccessible org-wide (SSO down). SEV2: Login issues affecting "
         "3+ users within 15 minutes. SEV3: Cosmetic or single-user issue.",
         ["portal", "severity", "runbook"]),
        ("RB-PRINT-007", "Print Services",
         "Severity criteria for Print Services incidents",
         "SEV1: All printers on all floors down. SEV2: All printers on a floor/department down, "
         "3+ correlated reports. SEV3: Single printer or single-user driver issue.",
         ["print", "severity", "runbook"]),
    ]
    for doc_id, service, summary, content, tags in runbooks:
        docs.append(KnowledgeBaseDocument(
            doc_id=doc_id, type="runbook", service=service,
            summary=summary, content=content, tags=tags,
        ))

    templates = [
        ("TPL-COMM-USER-001", "generic", "User-facing template: service disruption in progress",
         "We are aware that some users are experiencing issues with {service} since ~{time}. "
         "Our team is investigating. Next update by {next_update}. "
         "We apologize for the inconvenience.", "end_users",
         ["template", "user-facing", "in-progress"]),
        ("TPL-COMM-USER-002", "generic", "User-facing template: incident resolved",
         "The issue affecting {service} has been resolved as of {time}. "
         "If you continue to experience problems, please raise a new ticket. "
         "Thank you for your patience.", "end_users",
         ["template", "user-facing", "resolved"]),
        ("TPL-COMM-MGMT-001", "generic", "Management-facing template: incident declared",
         "{ticket_count} correlated tickets in {window} min, suspected cause: {suspected_cause}. "
         "Severity: {severity}. Status: being tracked. ETA next update: {next_update}.", "management",
         ["template", "management-facing", "in-progress"]),
        ("TPL-COMM-MGMT-002", "generic", "Management-facing template: incident resolved / closed",
         "Incident {incident_id} ({severity}) affecting {service} has been resolved. "
         "Root cause: {root_cause}. Total duration: {duration}. Post-mortem to follow.", "management",
         ["template", "management-facing", "resolved"]),
    ]
    for doc_id, service, summary, content, audience, tags in templates:
        docs.append(KnowledgeBaseDocument(
            doc_id=doc_id, type="communication_template", service=service,
            summary=summary, content=content, tags=tags, audience=audience,
        ))

    return docs


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    tickets, labels = generate_tickets()
    kb_docs = build_knowledge_base()

    jira_payload = {
        "total": len(tickets),
        "issues": [ticket_to_jira_issue(t) for t in tickets],
    }
    (OUT_TICKETS / "tickets.json").write_text(
        json.dumps(jira_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_TICKETS / "ground_truth.json").write_text(
        json.dumps([l.model_dump() for l in labels], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    by_type: dict[str, list[dict]] = {"post_mortem": [], "runbook": [], "communication_template": []}
    for d in kb_docs:
        by_type[d.type].append(d.model_dump(exclude_none=True))

    (OUT_KB / "historical_major_incidents.json").write_text(
        json.dumps(by_type["post_mortem"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_KB / "runbooks.json").write_text(
        json.dumps(by_type["runbook"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_KB / "communication_templates.json").write_text(
        json.dumps(by_type["communication_template"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_major = sum(1 for l in labels if l.is_major_incident_ticket)
    n_traps = sum(1 for l in labels if l.is_false_positive_trap)
    n_groups = len({l.incident_group_id for l in labels if l.incident_group_id})

    print(f"Tichete generate: {len(tickets)}")
    print(f"  - din care in burst-uri reale (major incident): {n_major}")
    print(f"  - din care false-positive traps: {n_traps}")
    print(f"  - din care izolate (zgomot): {len(tickets) - n_major - n_traps}")
    print(f"Grupuri reale de Major Incident (ground truth): {n_groups}")
    print(f"Documente knowledge base: {len(kb_docs)} "
          f"({len(by_type['post_mortem'])} post-mortems, "
          f"{len(by_type['runbook'])} runbooks, "
          f"{len(by_type['communication_template'])} comm. templates)")
    print(f"\nScrise in: {OUT_TICKETS} si {OUT_KB}")


if __name__ == "__main__":
    main()