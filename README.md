# IHouse — International Student Advisor

An AI-powered multi-agent advisor built for international university students. It covers the five domains that matter most when you first arrive: **visa & legal**, **academics**, **personal finance**, **jobs & internships**, and **taxes** — all in one conversation that remembers you across sessions.

Runs as a **Streamlit web app** (parchment-gold academia UI) or a **Rich CLI** — both backed by the same LangGraph pipeline.

Built for a hackathon by [AriS10223](https://github.com/AriS10223) and [anvipoluri](https://github.com/anvipoluri), developed in collaboration with **Claude (Anthropic)**.

---

## What it does

- **Personalized from session one** — onboarding captures your name, university, nationality, visa status, field of study, post-graduation plan, and time in the USA. Every agent uses this profile to give specific, not generic, advice.
- **Multi-agent routing** — an orchestrator reads your question and dispatches only the relevant specialist agents (1–3 per query). A synthesizer merges their answers into one response.
- **Web-grounded fact-checking** — hard claims (work-hour limits, form numbers, deadlines) are verified against authoritative sources before you see the answer:
  - Legal/visa claims → USCIS, DHS, State Dept only
  - Tax claims → IRS, SSA, Treasury only
  - Finance claims → CFPB, FDIC, Bankrate, NerdWallet
  - Academic questions → live search against your actual university's website
  - Jobs → live listings from Adzuna
- **Reflection loop** — a critic node evaluates the fact-checked answer on four criteria (disclaimer present, query addressed, consistency, unverified markers). If it fails, a revise node fixes only the flagged issues before the answer is shown.
- **Persistent profiles** — user profiles are stored in Supabase PostgreSQL and cached in Upstash Redis so every session picks up exactly where you left off — no re-onboarding.
- **Full observability** — every agent call, routing decision, fact-check, and search is traced in [W&B Weave](https://wandb.ai) with token counts and latency. Per-turn metrics (agents called, searches made, claims verified, critic pass/fail) are logged to a W&B dashboard table.

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM inference | [Groq](https://groq.com) — two-model split on LPU hardware (~600 tok/s) |
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Observability | [W&B Weave](https://wandb.ai/site/weave) + W&B Tables |
| Database | [Supabase](https://supabase.com) — PostgreSQL |
| Cache | [Upstash Redis](https://upstash.com) — serverless, HTTP-based |
| Web search | DuckDuckGo (no key required) |
| Job listings | [Adzuna API](https://developer.adzuna.com) (free tier) |
| Web UI | [Streamlit](https://streamlit.io) — parchment/academia theme, dark-mode resistant |
| CLI | [Rich](https://github.com/Textualize/rich) |

### Model split

Two Groq-hosted models are used — one for reasoning-heavy work, one for fast classification:

| Role | Model | Why |
|---|---|---|
| Domain agents | `meta-llama/llama-4-scout-17b-16e-instruct` | MoE architecture (17B active / 109B total params); best factual recall for visa law, tax rules, finance; 30k TPM free tier |
| Synthesizer | `meta-llama/llama-4-scout-17b-16e-instruct` | Needs coherence across multi-domain outputs |
| Fact-check + Revise | `meta-llama/llama-4-scout-17b-16e-instruct` | Claim extraction and web-grounded correction require strong reasoning |
| Router | `llama-3.1-8b-instant` | JSON classification only; 8B is sufficient and has 500k TPD free |
| Critic | `llama-3.1-8b-instant` | Pass/fail scoring on 4 criteria — minimal reasoning needed |

---

## Data layer

User profiles are stored in **Supabase PostgreSQL** and served through an **Upstash Redis** cache. This matters because each turn injects the profile into every agent's system prompt — that's 5–8 reads per question. Without a cache, each question would hammer the database; with Redis in front, Supabase is only touched on cold starts and profile updates.

```
Read path
─────────────────────────────────────────────────────────
App  →  Redis GET profile:{name}
              │
              ├─ HIT  ──────────────────────────────────→  return (no DB call)
              │
              └─ MISS  →  Supabase SELECT WHERE name=?
                                │
                                └─  populate Redis (TTL 24h)  →  return

Write path  (onboarding / "update profile" command)
─────────────────────────────────────────────────────────
App  →  Supabase UPSERT  (source of truth, always written first)
     →  Redis SET profile:{name}  (write-through, TTL reset to 24h)
```

**Failure handling:** If Redis is unavailable, `cache.py` logs a warning and falls through directly to Supabase — the app never crashes because of a cache failure. The database is the source of truth; Redis is an optimisation, not a dependency.

**Service role key:** The Python backend uses Supabase's service role key so all queries bypass Row Level Security. The key lives in `.env` server-side and is never exposed to clients. RLS enforcement is planned for the upcoming web frontend where a FastAPI backend will sit between the browser and Supabase.

### Profiles table

```sql
create table profiles (
    name            text primary key,
    university      text,
    nationality     text,
    visa_status     text,
    field_of_study  text,
    post_study_plan text,
    time_in_usa     text,
    updated_at      timestamptz default now()
);
```

---

## Observability

Every turn is traced end-to-end in W&B Weave and logged as a structured row in a W&B dashboard table.

### Weave trace tree

Each node in the pipeline is a named `@weave.op` with custom attributes attached via `weave.attributes()`, so the trace tree shows not just call structure but *why* each node did what it did:

```
run_turn
  └─ router              {"task": "routing", "available_domains": [...]}
  └─ agents_runner
       ├─ agent_legal    {"domain": "legal"}
       ├─ agent_academic {"domain": "academic", "search_type": "university_web", "university": "..."}
       └─ agent_jobs     {"domain": "jobs", "search_type": "job_listings", "keywords": "..."}
  └─ synthesizer
  └─ fact_check          {"claims_extracted": N, "route": [...], "search_count": N}
  └─ critic              {"task": "critique", "response_length": N}
  └─ revise              (only if critic fails)
```

### W&B metrics table

A `wandb.Table` named `turns` is rebuilt and logged after every query (W&B tables are immutable once logged — the rebuild-from-list pattern ensures the panel always shows the full session history):

| Column | What it captures |
|---|---|
| `turn` | Turn number in the session |
| `query` | The user's question (truncated to 200 chars) |
| `agents_called` | Which domain agents fired (e.g. `legal, tax`) |
| `route_reason` | The router's plain-English explanation for its choice |
| `num_searches` | Total web + job searches that turn triggered |
| `search_queries` | The actual strings sent to DuckDuckGo / Adzuna |
| `claims_found` | Number of verifiable claims extracted by the fact-checker |
| `critic_pass` | Whether the critic approved the answer on first pass |
| `revision_made` | Whether a revision pass was needed |
| `latency_ms` | Wall-clock time for the full turn |

Scalar metrics (`latency_ms`, `num_searches`, `claims_found`, `agents_called`) are also logged per turn for live line charts.

---

## Agent pipeline

```
user query
    │
    ▼
 router  ──── classifies query, picks 1–3 domain agents
    │
    ▼
agents_runner  ──── runs only selected agents
  ├─ agent_legal    (USCIS / DHS / travel.state.gov sources)
  ├─ agent_academic (live search against your university's website)
  ├─ agent_finance  (CFPB / FDIC / Bankrate sources)
  ├─ agent_jobs     (live Adzuna job listings)
  └─ agent_tax      (IRS / SSA / Treasury sources)
    │
    ▼
 synthesizer  ──── merges domain outputs into one coherent answer
    │
    ▼
 factcheck  ──── extracts hard claims, web-verifies against authoritative domains
    │
    ▼
 critic  ──── scores answer on 4 criteria
    ├─ pass ──→ shown to user
    └─ fail ──→ revise ──→ shown to user
```

---

## Web UI

`app.py` is a Streamlit frontend with a parchment-and-gold academia theme (dark-mode resistant via `.streamlit/config.toml`). It mirrors the three-page flow designed by [anvipoluri](https://github.com/anvipoluri/atlas-of-aid):

| Page | What it does |
|---|---|
| **Landing** | i-House hero — "First time here" or "Welcome back" |
| **Onboarding** | Collects name, university, nationality, visa status, field of study, post-study plan, time in USA — saves to Supabase + Redis |
| **Chat** | Full conversation interface — sticky-note message bubbles, domain icon rail, "Consulted:" agent attribution per response, profile sidebar |

The web UI calls the same `graph.invoke()` pipeline as the CLI — no separate API layer.

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/AriS10223/IHouse.git
cd IHouse
py -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt       # Mac/Linux
```

**2. Get API keys**

| Key | Where | Required |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Yes |
| `WANDB_API_KEY` | [wandb.ai](https://wandb.ai) → Settings → API keys | Yes |
| `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` | Supabase → Project Settings → API | Yes |
| `UPSTASH_REDIS_URL` + `UPSTASH_REDIS_TOKEN` | [console.upstash.com](https://console.upstash.com) | Yes |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | [developer.adzuna.com](https://developer.adzuna.com) | No — jobs agent degrades gracefully |

**3. Create the profiles table** in your Supabase SQL editor:
```sql
create table profiles (
    name            text primary key,
    university      text,
    nationality     text,
    visa_status     text,
    field_of_study  text,
    post_study_plan text,
    time_in_usa     text,
    updated_at      timestamptz default now()
);
```

**4. Configure**
```bash
cp .env.example .env
# fill in your keys
```

**5. Run**

Web UI (recommended):
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py   # Windows
# or: .venv/bin/python -m streamlit run app.py       # Mac/Linux
```
Then open `http://localhost:8501`.

CLI (alternative):
```bash
.\.venv\Scripts\python.exe main.py   # Windows
# or: .venv/bin/python main.py       # Mac/Linux
```

First run → onboarding form/questionnaire (profile saved to Supabase + cached in Redis). Return runs → profile loaded from Redis, straight into the conversation.

Web UI sidebar commands: **Update profile** · **New session**  
CLI in-session commands: `update profile` · `exit`

---

## Built with

This project was developed with [Claude Code](https://claude.ai/code) by Anthropic as the primary AI development assistant — architecture, all source code, prompt engineering, and iterative debugging were done in collaboration with Claude.
