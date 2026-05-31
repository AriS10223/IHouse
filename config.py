"""Central configuration — all cost/model knobs live here."""
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
WANDB_API_KEY: str = os.environ.get("WANDB_API_KEY", "")
ADZUNA_APP_ID: str = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY: str = os.environ.get("ADZUNA_APP_KEY", "")
WANDB_PROJECT: str = "intl-student-advisor"

# Groq models
# Scout (MoE 17B active / 109B total): best reasoning + knowledge, 30k TPM free tier
# 8b-instant: fast + high TPD (500k/day), ideal for simple classification tasks
FAST_MODEL: str = "llama-3.1-8b-instant"                          # router, critic — simple JSON tasks
STRONG_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"   # agents, synth, factcheck — reasoning + coherence

ROUTER_MODEL: str = FAST_MODEL
AGENT_MODEL: str = STRONG_MODEL
SYNTH_MODEL: str = STRONG_MODEL
FACTCHECK_MODEL: str = STRONG_MODEL   # extract claims + verify + revise
CRITIC_MODEL: str = FAST_MODEL        # pass/fail JSON on 4 criteria — 8b is sufficient

# Token limits — keep low for free tier
MAX_TOKENS: int = 1024       # default
AGENT_MAX_TOKENS: int = 700  # per specialist agent
SYNTH_MAX_TOKENS: int = 900
FACTCHECK_MAX_TOKENS: int = 1200

# Routing guards
MAX_AGENTS: int = 3  # cap fan-out; free tier ~200 req/day
FACT_CHECK_MAX_CLAIMS: int = 4  # claims extracted per response
FACT_CHECK_MAX_SEARCHES: int = 3  # DuckDuckGo lookups per turn

# All recognised domain keys
ALL_DOMAINS: list[str] = ["legal", "academic", "finance", "jobs", "tax"]

# Profile storage
PROFILES_FILE: str = "profiles.json"
