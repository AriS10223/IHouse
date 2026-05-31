# IHouse — International Student Advisor

An AI-powered multi-agent advisor built for international university students. It covers the five domains that matter most when you first arrive: **visa & legal**, **academics**, **personal finance**, **jobs & internships**, and **taxes** — all in one CLI conversation that remembers you across sessions.

Built for a hackathon by [AriS10223](https://github.com/AriS10223), developed in collaboration with **Claude (Anthropic)**.

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
- **Session memory** — conversation history persists across turns via LangGraph `MemorySaver`, so you can build on previous answers (e.g. refine a budget, follow up on visa questions).
- **Full observability** — every agent call, routing decision, and fact-check is traced in [W&B Weave](https://wandb.ai) with token counts and latency.

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM inference | [Groq](https://groq.com) — Llama 3.3 70B on LPU hardware (~600 tok/s) |
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Observability | [W&B Weave](https://wandb.ai/site/weave) |
| Web search | DuckDuckGo (no key required) |
| Job listings | [Adzuna API](https://developer.adzuna.com) (free tier) |
| CLI | [Rich](https://github.com/Textualize/rich) |

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

**2. Get API keys** (all free tiers, no credit card required)

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `WANDB_API_KEY` | [wandb.ai](https://wandb.ai) → Settings → API keys |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | [developer.adzuna.com](https://developer.adzuna.com) |

**3. Configure**
```bash
cp .env.example .env
# fill in your keys
```

**4. Run**
```bash
.\.venv\Scripts\python.exe main.py   # Windows
# or: .venv/bin/python main.py       # Mac/Linux
```

First run → 7-question onboarding. Return runs → picks up where you left off.

In-session commands: `update profile` · `exit`

---

## Agent pipeline

```
user query
    │
    ▼
 router  ──── classifies query, picks 1–3 agents
    │
    ▼
agents_runner  ──── runs only selected agents
  ├─ agent_legal    (USCIS / DHS sources)
  ├─ agent_academic (live university search)
  ├─ agent_finance  (CFPB / Bankrate sources)
  ├─ agent_jobs     (live Adzuna listings)
  └─ agent_tax      (IRS / SSA sources)
    │
    ▼
 synthesizer  ──── merges outputs into one answer
    │
    ▼
 factcheck  ──── web-verifies hard claims
    │
    ▼
 critic  ──── scores answer on 4 criteria
    ├─ pass ──→ shown to user
    └─ fail ──→ revise ──→ shown to user
```

---

## Built with

This project was developed with [Claude Code](https://claude.ai/code) by Anthropic as the primary AI development assistant — architecture, all source code, prompt engineering, and iterative debugging were done in collaboration with Claude.
