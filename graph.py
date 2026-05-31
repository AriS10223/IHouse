"""Build and compile the LangGraph advisor pipeline.

Flow:
  START → router → agents_runner → synthesizer → factcheck → END

MemorySaver checkpointer provides persistent session memory.
Pass config={"configurable": {"thread_id": <name>}} at invocation time
so each user gets their own conversation thread.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from orchestrator import router_node
from agents import agents_runner_node
from synthesizer import synthesizer_node
from factcheck import factcheck_node, critic_node, revise_node, should_revise


def build_graph():
    """Construct, compile, and return the advisor StateGraph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router", router_node)
    builder.add_node("agents_runner", agents_runner_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("factcheck", factcheck_node)
    builder.add_node("critic", critic_node)
    builder.add_node("revise", revise_node)

    # Linear pipeline up to factcheck
    builder.add_edge(START, "router")
    builder.add_edge("router", "agents_runner")
    builder.add_edge("agents_runner", "synthesizer")
    builder.add_edge("synthesizer", "factcheck")

    # Reflection loop: critic decides pass → END or fail → revise → END
    builder.add_edge("factcheck", "critic")
    builder.add_conditional_edges("critic", should_revise, {"revise": "revise", "__end__": END})
    builder.add_edge("revise", END)

    # Compile with MemorySaver for per-user persistent memory
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton — imported by main.py
advisor_graph = build_graph()
