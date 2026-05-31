"""Router / orchestrator node.

Classifies the user's query and decides which domain agents should answer.
Returns a JSON object {"agents": [...], "reason": "..."} — the reason field
becomes visible in the Weave trace, documenting WHY each agent fired.

Defensive parsing: if the LLM produces malformed JSON (common with free
models), we fall back to calling all agents rather than crashing.
"""
from __future__ import annotations

import json
import re

import weave

from config import ROUTER_MODEL, MAX_AGENTS, ALL_DOMAINS, MAX_TOKENS
from llm import chat
from onboarding import format_profile_for_prompt
from prompts import ROUTER_SYSTEM, ROUTER_USER
from state import AgentState


def _parse_route(raw: str) -> tuple[list[str], str]:
    """Extract (agents, reason) from the LLM's JSON output.

    Tries strict json.loads first, then falls back to a regex search,
    then returns all domains if both fail.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    # Try strict parse
    try:
        data = json.loads(clean)
        agents = [a for a in data.get("agents", []) if a in ALL_DOMAINS]
        reason = data.get("reason", "No reason provided.")
        if agents:
            return agents[:MAX_AGENTS], reason
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    try:
        agents_match = re.search(r'"agents"\s*:\s*\[([^\]]+)\]', clean)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', clean)
        if agents_match:
            raw_agents = [
                a.strip().strip('"') for a in agents_match.group(1).split(",")
            ]
            agents = [a for a in raw_agents if a in ALL_DOMAINS]
            reason = reason_match.group(1) if reason_match else "Parsed via regex fallback."
            if agents:
                return agents[:MAX_AGENTS], reason
    except Exception:
        pass

    # Hard fallback
    return ALL_DOMAINS[:MAX_AGENTS], "Router parsing failed — defaulting to broad coverage."


@weave.op(name="router")
def router_node(state: AgentState) -> dict:
    """LangGraph node: classify query and return routing decision."""
    query = state.get("query", "")
    profile = state.get("profile", {})
    profile_str = format_profile_for_prompt(profile)

    system = ROUTER_SYSTEM.format(profile_str=profile_str)
    user = ROUTER_USER.format(query=query)

    raw = chat(system=system, user=user, model=ROUTER_MODEL, max_tokens=256)
    agents, reason = _parse_route(raw)

    return {
        "route": agents,
        "route_reason": reason,
        # Reset per-turn fields so previous turn's data doesn't bleed through
        "agent_outputs": {},
        "draft": "",
        "final": "",
    }
