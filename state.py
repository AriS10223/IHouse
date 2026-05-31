"""LangGraph state definition for the multi-agent advisor graph."""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Persistent across turns (MemorySaver checkpointer) ──────────────
    messages: Annotated[list, add_messages]  # full conversation history

    # ── Set once at session start, carried through every turn ────────────
    profile: dict  # onboarding answers; read-only after first write

    # ── Reset each turn by the router node ──────────────────────────────
    query: str            # the current user question
    route: list[str]      # which domain agents to call
    route_reason: str     # router's human-readable explanation (visible in Weave)
    agent_outputs: dict   # domain → agent answer (overwritten each turn)
    draft: str            # synthesizer output
    final: str            # fact-checked final answer
    claims_found: int     # number of verifiable claims extracted by factcheck
    critic_pass: bool     # True = critic approved, False = needs revision
    critic_feedback: str  # critic's issues list, fed into the revise node
