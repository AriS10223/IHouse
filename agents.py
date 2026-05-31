"""Domain specialist agents — five named @weave.op functions for distinct traces.

legal / finance / tax  — shared impl: profile-aware LLM call with source restrictions
academic               — adds live university web search before the LLM call
jobs                   — adds live Adzuna job listings before the LLM call
"""
from __future__ import annotations

import weave

from config import AGENT_MODEL, AGENT_MAX_TOKENS
from llm import chat
from onboarding import format_profile_for_prompt
from prompts import (
    LEGAL_SYSTEM,
    ACADEMIC_SYSTEM,
    FINANCE_SYSTEM,
    JOBS_SYSTEM,
    TAX_SYSTEM,
)
from state import AgentState
from tools import (
    web_search,
    format_search_results,
    search_adzuna_jobs,
    format_job_listings,
)

# Domains that use the shared impl (no live search inside the agent itself)
_DOMAIN_PROMPTS: dict[str, str] = {
    "legal":   LEGAL_SYSTEM,
    "finance": FINANCE_SYSTEM,
    "tax":     TAX_SYSTEM,
}


def _build_history_str(state: AgentState, max_turns: int = 6) -> str:
    msgs = state.get("messages", [])[-max_turns:]
    lines = []
    for m in msgs:
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role, content = getattr(m, "type", ""), getattr(m, "content", "")
        if role in ("human", "user"):
            lines.append(f"User: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"Advisor: {content}")
    return "\n".join(lines) if lines else "(No prior conversation.)"


# ── Shared impl: legal / finance / tax ───────────────────────────────────────

def _run_domain_agent_impl(state: AgentState, domain: str) -> str:
    profile = state.get("profile", {})
    query = state.get("query", "")
    profile_str = format_profile_for_prompt(profile)
    history_str = _build_history_str(state)

    system = _DOMAIN_PROMPTS[domain].format(
        profile_str=profile_str,
        history_str=history_str,
    )
    return chat(system=system, user=query, model=AGENT_MODEL, max_tokens=AGENT_MAX_TOKENS)


# ── Academic impl: live university search ─────────────────────────────────────

def _run_academic_agent_impl(state: AgentState) -> str:
    profile = state.get("profile", {})
    query = state.get("query", "")
    university = profile.get("university", "")

    profile_str = format_profile_for_prompt(profile)
    history_str = _build_history_str(state)

    # Search the university's resources for this specific query
    if university and university.lower() not in ("unknown", "not specified"):
        search_query = f"{university} {query}"
        results = web_search(search_query, max_results=4)
        university_context = format_search_results(results)
    else:
        university_context = "(University not specified — providing general guidance.)"
        university = "your university"

    system = ACADEMIC_SYSTEM.format(
        profile_str=profile_str,
        history_str=history_str,
        university_name=university,
        university_context=university_context,
    )
    return chat(system=system, user=query, model=AGENT_MODEL, max_tokens=AGENT_MAX_TOKENS)


# ── Jobs impl: live Adzuna listings ──────────────────────────────────────────

def _run_jobs_agent_impl(state: AgentState) -> str:
    profile = state.get("profile", {})
    query = state.get("query", "")
    field = profile.get("field_of_study", "")
    post_plan = profile.get("post_study_plan", "")

    profile_str = format_profile_for_prompt(profile)
    history_str = _build_history_str(state)

    # Build a targeted Adzuna keyword from field + intent
    if "intern" in query.lower():
        keywords = f"{field} internship" if field else "internship"
    elif "job" in query.lower() or "work" in query.lower():
        keywords = f"{field} entry level" if field else "entry level"
    else:
        keywords = f"{field} internship OR entry level" if field else "internship"

    jobs = search_adzuna_jobs(keywords=keywords, max_results=5)
    job_listings = format_job_listings(jobs)

    system = JOBS_SYSTEM.format(
        profile_str=profile_str,
        history_str=history_str,
        job_listings=job_listings,
    )
    return chat(system=system, user=query, model=AGENT_MODEL, max_tokens=AGENT_MAX_TOKENS)


# ── Named @weave.op wrappers (distinct trace names) ──────────────────────────

@weave.op(name="agent_legal")
def run_legal(state: AgentState) -> str:
    return _run_domain_agent_impl(state, "legal")


@weave.op(name="agent_academic")
def run_academic(state: AgentState) -> str:
    return _run_academic_agent_impl(state)


@weave.op(name="agent_finance")
def run_finance(state: AgentState) -> str:
    return _run_domain_agent_impl(state, "finance")


@weave.op(name="agent_jobs")
def run_jobs(state: AgentState) -> str:
    return _run_jobs_agent_impl(state)


@weave.op(name="agent_tax")
def run_tax(state: AgentState) -> str:
    return _run_domain_agent_impl(state, "tax")


AGENT_REGISTRY: dict[str, callable] = {
    "legal":    run_legal,
    "academic": run_academic,
    "finance":  run_finance,
    "jobs":     run_jobs,
    "tax":      run_tax,
}


@weave.op(name="agents_runner")
def agents_runner_node(state: AgentState) -> dict:
    route = state.get("route", [])
    outputs: dict[str, str] = {}
    for domain in route:
        if domain in AGENT_REGISTRY:
            outputs[domain] = AGENT_REGISTRY[domain](state)
        else:
            outputs[domain] = f"[Unknown domain: {domain}]"
    return {"agent_outputs": outputs}
