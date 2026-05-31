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

# Groq models — all free tier, LPU-accelerated (~600 tok/s)
FAST_MODEL: str = "llama-3.3-70b-versatile"   # router + agents
STRONG_MODEL: str = "llama-3.3-70b-versatile"  # synth + factcheck

ROUTER_MODEL: str = FAST_MODEL
AGENT_MODEL: str = FAST_MODEL
SYNTH_MODEL: str = STRONG_MODEL
FACTCHECK_MODEL: str = STRONG_MODEL

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
