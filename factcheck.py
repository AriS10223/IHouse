"""Fact-check, critic, and revise nodes.

Pipeline per turn:
  factcheck → critic → END          (if critic passes)
  factcheck → critic → revise → END (if critic finds issues)

factcheck: extract hard claims → DuckDuckGo verify → correct draft + add disclaimers.
critic:    evaluate final on 4 criteria (disclaimer, query addressed, consistency, unverified).
revise:    fix only the issues the critic flagged — one revision pass, no second critique.
"""
from __future__ import annotations

import json
import re

import weave

from config import FACTCHECK_MODEL, FACTCHECK_MAX_TOKENS, FACT_CHECK_MAX_CLAIMS, FACT_CHECK_MAX_SEARCHES
from llm import chat
from prompts import (
    FACTCHECK_EXTRACT_SYSTEM,
    FACTCHECK_VERIFY_SYSTEM,
    FACTCHECK_VERIFY_USER,
    CRITIC_SYSTEM,
    CRITIC_USER,
    REVISE_SYSTEM,
    REVISE_USER,
)
from state import AgentState
from tools import web_search, web_search_authoritative, format_search_results, DOMAIN_SEARCH_RESTRICTIONS

_FALLBACK_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This is informational only — not legal, tax, or financial advice.\n"
    "For immigration matters, consult your DSO/ISSO. For taxes, use GLACIER/Sprintax or a CPA.\n"
    "For financial decisions, consult a licensed advisor."
)


def _extract_claims(draft: str) -> list[str]:
    """Ask the LLM to extract verifiable factual claims from the draft."""
    system = FACTCHECK_EXTRACT_SYSTEM.format(max_claims=FACT_CHECK_MAX_CLAIMS)
    raw = chat(
        system=system,
        user=f"Advisor response:\n{draft}",
        model=FACTCHECK_MODEL,
        max_tokens=400,
    )
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        claims = json.loads(clean)
        if isinstance(claims, list):
            return [str(c) for c in claims[:FACT_CHECK_MAX_CLAIMS]]
    except json.JSONDecodeError:
        pass
    # Fallback: try to extract quoted strings
    return re.findall(r'"([^"]{10,})"', clean)[:FACT_CHECK_MAX_CLAIMS]


def _gather_evidence(claims: list[str], route: list[str] | None = None) -> str:
    """Search for evidence per claim, restricted to authoritative domains where applicable.

    legal/tax claims → government sites only (.gov)
    finance claims   → reputable finance sites
    academic/jobs    → unrestricted (those agents already ran live searches)
    """
    # Build the combined domain restriction from all agents that ran
    restricted_domains: list[str] = []
    for domain in (route or []):
        restricted_domains.extend(DOMAIN_SEARCH_RESTRICTIONS.get(domain, []))

    evidence_blocks = []
    for i, claim in enumerate(claims[:FACT_CHECK_MAX_SEARCHES]):
        if restricted_domains:
            results = web_search_authoritative(claim, restricted_domains, max_results=3)
        else:
            results = web_search(claim, max_results=3)
        snippet = format_search_results(results)
        evidence_blocks.append(f"Claim {i+1}: {claim}\nEvidence:\n{snippet}")
    return "\n\n".join(evidence_blocks) if evidence_blocks else "(No web evidence gathered.)"


@weave.op(name="fact_check")
def factcheck_node(state: AgentState) -> dict:
    """LangGraph node: extract claims, web-verify, correct draft, add disclaimers."""
    draft = state.get("draft", "")

    if not draft.strip():
        return {"final": "(No response generated.)" + _FALLBACK_DISCLAIMER, "claims_found": 0}

    # Step 1: extract verifiable claims
    try:
        claims = _extract_claims(draft)
    except Exception as exc:
        print(f"[factcheck] Claim extraction failed: {exc}")
        return {"final": draft + _FALLBACK_DISCLAIMER, "claims_found": 0}

    if not claims:
        # Nothing hard to verify — just add disclaimers
        return {"final": draft + _FALLBACK_DISCLAIMER, "claims_found": 0}

    # Step 2: gather web evidence (route-aware domain restrictions)
    route = state.get("route", [])
    try:
        with weave.attributes({"claims_extracted": len(claims), "route": route,
                               "search_count": min(len(claims), FACT_CHECK_MAX_SEARCHES)}):
            evidence_str = _gather_evidence(claims, route=route)
    except Exception as exc:
        print(f"[factcheck] Evidence gathering failed: {exc}")
        return {"final": draft + _FALLBACK_DISCLAIMER, "claims_found": len(claims)}

    # Step 3: verify and correct
    try:
        user_prompt = FACTCHECK_VERIFY_USER.format(
            draft=draft,
            evidence_str=evidence_str,
        )
        final = chat(
            system=FACTCHECK_VERIFY_SYSTEM,
            user=user_prompt,
            model=FACTCHECK_MODEL,
            max_tokens=FACTCHECK_MAX_TOKENS,
        )
        return {"final": final.strip(), "claims_found": len(claims)}
    except Exception as exc:
        print(f"[factcheck] Verification LLM call failed: {exc}")
        return {"final": draft + _FALLBACK_DISCLAIMER, "claims_found": len(claims)}


# ── Critic node ───────────────────────────────────────────────────────────────

@weave.op(name="critic")
def critic_node(state: AgentState) -> dict:
    """Evaluate the fact-checked answer on 4 criteria. Sets critic_pass and critic_feedback."""
    final = state.get("final", "")
    query = state.get("query", "")

    try:
        with weave.attributes({"task": "critique", "response_length": len(final)}):
            raw = chat(
                system=CRITIC_SYSTEM,
                user=CRITIC_USER.format(query=query, final=final),
                model=FACTCHECK_MODEL,
                max_tokens=300,
            )
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        data = json.loads(clean)
        if data.get("pass") is True:
            return {"critic_pass": True, "critic_feedback": ""}
        issues = data.get("issues", ["Unspecified issues."])
        return {"critic_pass": False, "critic_feedback": "\n".join(f"- {i}" for i in issues)}
    except Exception as exc:
        print(f"[critic] Evaluation failed ({exc}) — passing through.")
        return {"critic_pass": True, "critic_feedback": ""}


def should_revise(state: AgentState) -> str:
    """Conditional edge: route to 'revise' if critic failed, otherwise END."""
    return "revise" if not state.get("critic_pass", True) else "__end__"


# ── Revise node ───────────────────────────────────────────────────────────────

@weave.op(name="revise")
def revise_node(state: AgentState) -> dict:
    """Fix only the issues the critic flagged. One pass — no second critique."""
    final = state.get("final", "")
    critic_feedback = state.get("critic_feedback", "")

    try:
        revised = chat(
            system=REVISE_SYSTEM,
            user=REVISE_USER.format(final=final, critic_feedback=critic_feedback),
            model=FACTCHECK_MODEL,
            max_tokens=FACTCHECK_MAX_TOKENS,
        )
        return {"final": revised.strip()}
    except Exception as exc:
        print(f"[revise] Revision failed ({exc}) — keeping original.")
        return {"final": final}
