"""Web search and job search helpers.

web_search()              — general DuckDuckGo search (no key, free)
web_search_authoritative()— DuckDuckGo restricted to specific domains
search_adzuna_jobs()      — Adzuna job listings API (free, needs APP_ID + APP_KEY)
format_search_results()   — DuckDuckGo results → readable text block for LLM
format_job_listings()     — Adzuna results → readable text block for LLM
"""
from __future__ import annotations

import os

# ── Turn-scoped search tracker ────────────────────────────────────────────────
# Reset at the start of each turn in main.py; read after the turn completes.

_turn_searches: list[dict] = []


def reset_turn_tracking() -> None:
    global _turn_searches
    _turn_searches = []


def get_turn_searches() -> list[dict]:
    """Return searches made during the current turn (web + jobs)."""
    return list(_turn_searches)


# ── Authoritative domain lists per subject area ───────────────────────────────

LEGAL_DOMAINS = [
    "uscis.gov", "dhs.gov", "travel.state.gov",
    "ice.gov", "studyinthestates.dhs.gov", "nafsa.org",
]

TAX_DOMAINS = [
    "irs.gov", "ssa.gov", "treasury.gov",
]

FINANCE_DOMAINS = [
    "consumerfinance.gov", "fdic.gov", "investor.gov",
    "bankrate.com", "nerdwallet.com", "investopedia.com",
]

# Maps route domain keys to their search restriction lists
DOMAIN_SEARCH_RESTRICTIONS: dict[str, list[str]] = {
    "legal":   LEGAL_DOMAINS,
    "tax":     TAX_DOMAINS,
    "finance": FINANCE_DOMAINS,
}


# ── DuckDuckGo helpers ────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> list[dict]:
    """General DuckDuckGo search. Degrades gracefully on failure."""
    _turn_searches.append({"type": "web", "query": query})
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=max_results)
        return list(results) if results else []
    except Exception as exc:
        print(f"[tools] web_search failed for '{query}': {exc}")
        return []


def web_search_authoritative(
    query: str, domains: list[str], max_results: int = 3
) -> list[dict]:
    """DuckDuckGo search biased toward specific domains via site: filters.

    Falls back to unrestricted search if the restricted query returns nothing,
    so a network hiccup or niche claim never silently drops evidence.
    """
    site_filter = " OR ".join(f"site:{d}" for d in domains)
    results = web_search(f"({site_filter}) {query}", max_results=max_results)
    if not results:
        results = web_search(query, max_results=max_results)
    return results


def format_search_results(results: list[dict]) -> str:
    """Turn DuckDuckGo results into a readable block for the LLM."""
    if not results:
        return "(No search results found.)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"[{i}] {title}\n{body}\nSource: {href}")
    return "\n\n".join(lines)


# ── Adzuna job search ─────────────────────────────────────────────────────────

def search_adzuna_jobs(
    keywords: str,
    location: str = "",
    country: str = "us",
    max_results: int = 5,
) -> list[dict]:
    """Search Adzuna for job/internship listings.

    Returns raw Adzuna result dicts. Returns [] if keys are missing or the
    request fails — jobs agent degrades to training-knowledge advice only.
    Register free at developer.adzuna.com to get APP_ID and APP_KEY.
    """
    _turn_searches.append({"type": "jobs", "query": keywords})
    app_id = os.environ.get("ADZUNA_APP_ID", "")
    app_key = os.environ.get("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        return []

    try:
        import requests
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": keywords,
            "results_per_page": max_results,
            "content-type": "application/json",
        }
        if location:
            params["where"] = location
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        print(f"[tools] Adzuna search failed for '{keywords}': {exc}")
        return []


def format_job_listings(jobs: list[dict]) -> str:
    """Turn Adzuna results into a readable block for the jobs agent LLM prompt."""
    if not jobs:
        return "(No live job listings found — providing general guidance based on current market knowledge.)"
    lines = []
    for j in jobs:
        title = j.get("title", "Unknown Role")
        company = j.get("company", {}).get("display_name", "Unknown Company")
        location = j.get("location", {}).get("display_name", "")
        description = (j.get("description", "") or "")[:200].strip()
        url = j.get("redirect_url", "")
        lines.append(
            f"• {title} — {company} ({location})\n"
            f"  {description}{'...' if description else ''}\n"
            f"  Apply: {url}"
        )
    return "\n\n".join(lines)
