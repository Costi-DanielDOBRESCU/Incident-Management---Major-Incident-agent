# Documentație Proiect
## Agent AI pentru Detecția Incidentelor Majore și Generarea Comunicării Automate
### Categorie: IT Service Management (ITSM) – Incident Management

**Denumire agent:** Major Incident Agent (MIA)
**Autor:** _[Daniel Dobrescu]_

---

## Cuprins

1. [Definirea problemei și scopul proiectului](#1-definirea-problemei-și-scopul-proiectului)
2. [Înțelegerea procesului (AS-IS)](#2-înțelegerea-procesului-as-is)
3. [Soluția propusă / Fluxul TO-BE](#3-soluția-propusă--fluxul-to-be)
4. [Arhitectura generală a sistemului](#4-arhitectura-generală-a-sistemului)
5. [Structura datelor și abordarea RAG](#5-structura-datelor-și-abordarea-rag)
6. [Reasoning / Decizie / Execuție](#6-reasoning--decizie--execuție)
7. [KPI-uri și Success Criteria](#7-kpi-uri-și-success-criteria)
8. [Observabilitate și audit](#8-observabilitate-și-audit)
9. [Stack tehnic și livrarea ca produs](#9-stack-tehnic-și-livrarea-ca-produs)
10. [Riscuri, limitări și dezvoltări viitoare](#10-riscuri-limitări-și-dezvoltări-viitoare)
11. [Anexe](#11-anexe)

---

## 1. Definirea problemei și scopul proiectului
### 1.1 Context business / IT

În organizațiile mari, un incident major — precum indisponibilitatea unei aplicații critice, o defecțiune a rețelei sau o cădere a serviciilor cloud — generează frecvent un val de tichete individuale, raportate într-un interval scurt de timp de utilizatori diferiți, dar având aceeași cauză reală.

Într-un proces ITSM tradițional, fiecare incident este înregistrat și analizat separat de către echipa de suport. Atunci când mai mulți utilizatori raportează aceeași problemă, tichetele similare pot fi tratate inițial ca incidente distincte. Identificarea faptului că acestea au o cauză comună și reprezintă, de fapt, **un singur incident major**, se bazează în mare parte pe analiza manuală și pe experiența operatorilor sau a Incident Managerului.

Corelarea incidentelor presupune compararea unor informații precum titlul și descrierea, serviciul sau aplicația afectată, utilizatorii, locația și momentul raportării. Atunci când numărul de tichete este mare, analiza manuală devine dificilă și poate întârzia identificarea unui incident major. 

După identificarea unui posibil incident major, comunicarea către utilizatorii afectați și management este adesea realizată manual. Acest lucru poate duce la întârzieri și la mesaje diferite ca exprimare, structură și detaliile date, în funcție de destinatari.

Această abordare poate conduce la:
- **identificarea întârziată a incidentelor majore, din cauza lipsei corelării automate între tichetele similare**;
- **creșterea efortului manual, necesar pentru analizarea unui număr mare de tichete**;
- **prioritizare și escaladare inconsistente, în funcție de experiența și disponibilitatea operatorilor**;
- **creșterea numărului de tichete duplicate, deoarece utilizatorii pot raporta aceeași problemă fără să știe că aceasta este deja investigată**;
- **întârzierea comunicării către utilizatori și management, precum și transmiterea unor mesaje diferite ca structură și conținut**;
- **creșterea MTTR (Mean Time to Repair), ca urmare a timpului suplimentar necesar pentru identificarea și gestionarea incidentului**;
- **scăderea încrederii utilizatorilor și a business-ului în serviciile IT, ca urmare a lipsei de vizibilitate și a comunicării întârziate**.

Prin urmare, problema nu este doar volumul mare de tichete, ci și dificultatea de a identifica rapid faptul că mai multe incidente aparent independente au aceeași cauză și fac parte dintr-un singur incident major.

### 1.2 Obiectivul proiectului

Obiectivul proiectului este dezvoltarea unui agent AI (**Major Incident Agent**) care:
1. **Detectează automat** posibile incidente majore prin analiza similarității semantice (embeddings + cosine similarity + clustering) între tichetele nou create într-un interval de timp definit.
2. **Evaluează** (prin LLM, cu output structurat/validat) dacă un cluster de tichete similare reprezintă cu adevărat un candidat de incident major, pe baza informațiilor istorice relevante (RAG).
3. **Generează automat** un draft de comunicare (separat pentru utilizatorii finali și pentru management), pe baza unor template-uri și a informațiilor relevante despre incident.
4. Trece comunicarea și declararea incidentului major printr-un **punct de aprobare umană** (human-in-the-loop) înainte de trimiterea rezultatului.
5. **Loghează și expune** întregul lanț de decizie (audit + observabilitate).

### 1.3 Scopul proiectului (In-Scope)

- Introducerea de tichete (mock data, format text – „problem report”) simulând un flux realist dintr-un ITSM (ex. ServiceNow/Jira-like).
- Calcul embeddings pe descrierea tichetelor + clustering pe similaritate cosinus, într-o fereastră de timp configurabilă (ex. ultimele 15-30 min / ultimele N tichete).
- Un modul LLM de **assessment** (confirmă / infirmă candidatul de incident major, motivează, estimează gravitatea) – output JSON validat Pydantic.
- Un modul RAG (ChromaDB) cu bază de cunoștințe: incidente majore istorice + post-mortem-uri + runbook-uri + template-uri de comunicare, folosit pentru a fundamenta atât decizia, cât și textul comunicării (cu citări).
- Generare comunicare (2 variante: user-facing, management-facing), output JSON validat.
- Guardrail de aprobare umană înainte de orice acțiune „vizibilă extern" (trimitere comunicare, declarare oficială major incident).
- Execuție deterministă a acțiunilor aprobate (tool-uri: trimitere notificare, update status tichete, link-uire tichete la incidentul părinte).
- Observabilitate end-to-end (Arize Phoenix) și audit trail.
- KPI de măsurare a impactului față de procesul manual.
- Livrare: API FastAPI + UI Streamlit + containerizare Docker.

### 1.4 În afara scopului (Out of Scope)

- Remedierea tehnică efectivă a incidentului (root cause fixing) – agentul **nu** execută acțiuni de remediere infrastructură (restart servicii, rollback deploy etc.), doar detectează, evaluează și comunică.
- Integrare live cu un ITSM real de producție (ServiceNow/Jira) – se lucrează cu date mock și, opțional, conectori simulați.
- Traducere în mai multe limbi a comunicărilor (se livrează în limba definită în configurare, implicit EN).
- Predicție proactivă de incidente înainte ca tichetele să apară (capacity/anomaly forecasting) – agentul e reactiv, bazat pe tichetele deja create.
- Fine-tuning propriu al unui model LLM (se folosește few-shot / prompt engineering pe model open-source via Ollama sau Groq free tier).

### 1.5 Asumpții (Assumptions)

- Tichetele conțin minim: id, timestamp, textul problemei (free text), serviciul/aplicația afectată (dacă e cunoscută), prioritate inițială, reporter.
- Volumul și distribuția temporală a tichetelor mock reflectă tipare realiste (perioade „calme" + „burst"-uri simulând incidente majore).
- Un incident „major" este definit organizațional printr-un set de criterii cuantificabile, nu doar prin percepția LLM-ului.
- Accesul la modelele LLM/embeddings (Ollama local sau Groq API free tier) este disponibil în mediul de dezvoltare/demo.

## 2. Înțelegerea procesului (AS-IS)
## 3. Soluția propusă / Fluxul TO-BE
## 4. Arhitectura generală a sistemului
## 5. Structura datelor și abordarea RAG
## 6. Reasoning / Decizie / Execuție
## 7. KPI-uri și Success Criteria
## 8. Observabilitate și audit
## 9. Stack tehnic și livrarea ca produs
## 10. Riscuri, limitări și dezvoltări viitoare
## 11. Anexe

---

Licență
Acest proiect este licențiat sub licența MIT. Bibliotecile și resursele terțe rămân supuse propriilor licențe.

---
