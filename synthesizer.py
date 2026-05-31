"""Synthesizer node — merges domain agent outputs into one coherent answer."""
from __future__ import annotations

import weave

from config import SYNTH_MODEL, SYNTH_MAX_TOKENS
from llm import chat
from prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER
from state import AgentState


def _format_agent_outputs(agent_outputs: dict[str, str]) -> str:
    """Render agent outputs as a labelled block for the synthesizer prompt."""
    lines = []
    for domain, text in agent_outputs.items():
        label = domain.upper()
        lines.append(f"=== {label} AGENT ===\n{text.strip()}")
    return "\n\n".join(lines) if lines else "(No agent outputs.)"


@weave.op(name="synthesizer")
def synthesizer_node(state: AgentState) -> dict:
    """LangGraph node: merge agent outputs into a single coherent draft."""
    query = state.get("query", "")
    agent_outputs = state.get("agent_outputs", {})

    agent_outputs_str = _format_agent_outputs(agent_outputs)

    user_prompt = SYNTHESIZER_USER.format(
        query=query,
        agent_outputs_str=agent_outputs_str,
    )

    draft = chat(
        system=SYNTHESIZER_SYSTEM,
        user=user_prompt,
        model=SYNTH_MODEL,
        max_tokens=SYNTH_MAX_TOKENS,
    )

    return {"draft": draft.strip()}
