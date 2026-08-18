# Agentic AI Interview Manager

A multi-agent AI system that manages the full hiring interview pipeline — from resume screening to final hire decisions — with human-in-the-loop checkpoints and full audit logging.

---

## What it does

1. **Resume Upload & Parsing** — Upload multiple PDF resumes and a job description through the Streamlit UI
2. **AI Screening Agent** — GPT-4o scores every candidate (0–100) with strengths, weaknesses, and a recommendation, enriched by RAG context from a ChromaDB vector database
3. **Human Checkpoint 1** — Recruiter reviews AI scores, edits them, selects shortlist, or rejects all and re-screens
4. **Interview Question Agent** — GPT-4o generates 8 tailored questions per shortlisted candidate (3 technical, 3 behavioral, 2 situational)
5. **Human Checkpoint 2** — Recruiter edits questions and makes final HIRE/REJECT decisions
6. **Audit Logging** — Every AI decision and human override is persisted to SQLite for full auditability

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (ui/app.py)                    │
│   Page 1: Setup  │  Page 2: Screening  │  Page 3: Questions     │
│                  │  [Checkpoint 1]     │  [Checkpoint 2]        │
└────────────────────────────┬────────────────────────────────────┘
                             │  st.session_state
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                           │
│                                                                 │
│   START                                                         │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐    ┌──────────────┐                       │
│  │ screening_agent  │───▶│ checkpoint_1 │◀── interrupt_before   │
│  │   (GPT-4o + RAG) │    │   [PAUSED]   │                       │
│  └──────────────────┘    └──────┬───────┘                       │
│           ▲                     │ APPROVED/EDITED               │
│           │ REJECTED            ▼                               │
│           │             ┌──────────────────┐                    │
│           └─────────────│  question_agent  │                    │
│                         │  (GPT-4o + RAG)  │                    │
│                         └────────┬─────────┘                    │
│                                  │                              │
│                                  ▼                              │
│                         ┌──────────────┐                        │
│                         │ checkpoint_2 │◀── interrupt_before    │
│                         │   [PAUSED]   │                        │
│                         └──────┬───────┘                        │
│                                │                                │
│                                ▼                                │
│                         ┌──────────────┐                        │
│                         │  log_agent   │──▶ SQLite              │
│                         └──────────────┘                        │
│                                │                                │
│                               END                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       RAG Layer (ChromaDB)                      │
│  Collection: resumes          (opensporks/resumes)              │
│  Collection: job_descriptions (NxtGenIntern/...)                │
│  Collection: interview_questions (ali-alkhars/interviews)       │
│                                                                 │
│  Embeddings: sentence-transformers all-MiniLM-L6-v2             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### 1. Clone / navigate to the project

```bash
cd interview_manager
```

### 2. Create a virtual environment (recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Edit `.env`:

```
OPENAI_API_KEY=sk-...your-key-here...
CHROMA_PERSIST_DIR=./chroma_db
SQLITE_DB_PATH=./database/interview_log.db
```

### 5. Run the application

```bash
python main.py
```

On first run, this will:
- Initialize the SQLite database
- Download and index 3 HuggingFace datasets into ChromaDB (~5–10 minutes)
- Launch the Streamlit UI at http://localhost:8501

Subsequent runs skip dataset loading (ChromaDB is persisted to disk).

---

## How to Use (Step by Step)

### Step 1 — Setup Page
1. Open the app at http://localhost:8501
2. Paste the full job description into the text area
3. Upload one or more PDF resumes using the file uploader
4. Click **"Start Interview Pipeline"**

### Step 2 — Screening Results (Checkpoint 1)
- The pipeline runs the screening agent and pauses automatically
- Review each candidate's AI score, reasoning, strengths, and weaknesses
- Check/uncheck candidates to include in the shortlist
- Edit scores if desired (creates an audit trail entry)
- Click **"Approve Shortlist"** to proceed, or **"Reject All & Re-screen"** to restart

### Step 3 — Interview Questions (Checkpoint 2)
- The pipeline generates 8 tailored questions per shortlisted candidate
- Review questions; edit any text directly in the UI
- Toggle **HIRE / REJECT** for each candidate
- Click **"Finalize & Log Decisions"**

### Step 4 — Audit Log
- View all AI decisions and human overrides in a filterable table
- Filter by candidate name
- Download a CSV export
- See summary stats: screened / shortlisted / hired / overrides

---

## Dataset Sources

| Dataset | HuggingFace ID | Used For |
|---------|---------------|----------|
| Resumes | `opensporks/resumes` | Resume RAG context (first 300 rows) |
| Job Descriptions | `NxtGenIntern/job_titles_and_descriptions` | Job profile benchmarks (all rows) |
| Interview Questions | `ali-alkhars/interviews` | Interview Q&A retrieval (first 200 rows) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph 0.3.x |
| LLM | OpenAI GPT-4o |
| LLM Framework | LangChain 0.3.x |
| Vector Database | ChromaDB 0.6.x |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Frontend | Streamlit 1.44.x |
| Audit Database | SQLite (via Python stdlib) |
| PDF Parsing | PyPDF2 3.x |
| Dataset Loading | HuggingFace datasets 3.x |
| Environment | python-dotenv |

---

## Project Structure

```
interview_manager/
├── .env                          # API keys
├── requirements.txt              # pinned dependencies
├── README.md                     # this file
├── main.py                       # entry point
│
├── data/
│   └── load_datasets.py          # HuggingFace → ChromaDB loader
│
├── agents/
│   ├── state.py                  # InterviewState TypedDict
│   ├── screening_agent.py        # Agent 1: score and rank resumes
│   ├── question_agent.py         # Agent 2: generate interview questions
│   └── log_agent.py              # logs to SQLite
│
├── graph/
│   └── workflow.py               # LangGraph StateGraph
│
├── rag/
│   ├── vectorstore.py            # ChromaDB collections + query
│   └── embeddings.py             # sentence-transformer embedding fn
│
├── human_checkpoints/
│   └── checkpoints.py            # checkpoint node functions
│
├── database/
│   └── override_log.py           # SQLite schema + CRUD
│
└── ui/
    └── app.py                    # Streamlit multi-page UI
```
