# Major Incident Agent (MIA)

Agent AI pentru detecția incidentelor majore prin clustering pe embeddings +
generarea automată a comunicării (user-facing / management), cu human-in-the-loop
și observabilitate completă.

## Structură proiect

```
major_incident_agent/
├── app/
│   ├── config.py              # setari centralizate (praguri, provideri LLM)
│   ├── models/                # scheme Pydantic (Ticket, IncidentAssessment, ...)
│   ├── data/                  # date mock (tichete, knowledge base) + chroma_store
│   ├── ingestion/              # mock Jira-like API + adapter fetch_tickets
│   ├── detection/              # embeddings + clustering (determinist)
│   ├── rag/                    # ChromaDB ingestion + query_knowledge_base + RAGAS
│   ├── agents/                 # Assessment Agent, Communication Agent (LLM)
│   ├── orchestrator/           # LangGraph state machine + gate-uri aprobare
│   ├── execution/              # tool-uri deterministe + audit log
│   ├── observability/          # instrumentare Arize Phoenix
│   ├── api/                    # FastAPI
│   └── ui/                     # Streamlit
├── tests/                      # pytest
├── scripts/                    # scripturi utilitare (generare date mock, etc.)
├── requirements.txt
├── .env.example
└── docker-compose.yml          # (adaugat in etapa de containerizare)
```

## Setup local (Windows / VS Code, Python 3.10)

```powershell
# 1. creeaza mediul virtual
python -m venv venv

# 2. activeaza-l
venv\Scripts\activate

# 3. instaleaza dependintele
pip install --upgrade pip
pip install -r requirements.txt

# 4. copiaza fisierul de configurare
copy .env.example .env
```

## Setup local (macOS / Linux)

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Ollama (LLM local, provider principal)

```bash
# instaleaza Ollama (https://ollama.com/download), apoi:
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve   # daca nu porneste automat
```

## Groq (fallback, free tier)

Creează un API key gratuit pe https://console.groq.com și pune-l în `.env`
la `GROQ_API_KEY`. Se folosește automat doar dacă Ollama nu răspunde
(rate-limit / timeout / indisponibil) — vezi `app/config.py`.

## Verificare rapidă setup

```bash
python -m app.config
```

Ar trebui să afișeze dict-ul de settings (praguri, provideri) fără erori.

## Status implementare (checklist etape)

- [x] Etapa 0 — Setup proiect (structură, venv, requirements, config)
- [ ] Etapa 1 — Date mock (tichete + knowledge base) + mock Jira-like API
- [ ] Etapa 2 — Modele Pydantic
- [ ] Etapa 3 — Detection Pipeline (embeddings + clustering)
- [ ] Etapa 4 — RAG cu ChromaDB + evaluare RAGAS
- [ ] Etapa 5 — Assessment Agent
- [ ] Etapa 6 — Communication Agent
- [ ] Etapa 7 — Orchestrator (LangGraph) + human-in-the-loop
- [ ] Etapa 8 — Execution Layer + audit log
- [ ] Etapa 9 — Observabilitate (Arize Phoenix)
- [ ] Etapa 10 — API FastAPI
- [ ] Etapa 11 — UI Streamlit
- [ ] Etapa 12 — KPI & comparație baseline vs. agentic
- [ ] Etapa 13 — Dockerizare
- [ ] Etapa 14 — Teste pytest
