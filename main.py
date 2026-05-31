"""Entry point — CLI chat loop for the International Student Advisor.

Startup sequence:
  1. load_dotenv() — read .env
  2. weave.init()  — MUST happen before the OpenAI client is created
  3. Build LLM client (lazily in llm.py, so this is fine)
  4. Onboarding — load or create user profile
  5. Build/import the advisor graph
  6. CLI loop with rich formatting

Commands:
  exit / quit      — end the session
  update profile   — re-run the onboarding questionnaire
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# ── 1. Load .env FIRST ────────────────────────────────────────────────────────
load_dotenv()

# Validate required keys before going further
_missing = [k for k in ("GROQ_API_KEY", "WANDB_API_KEY") if not os.environ.get(k)]
if _missing:
    print(
        f"\n[ERROR] Missing environment variables: {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in your API keys.\n"
        "  Groq      (free): https://console.groq.com\n"
        "  W&B Weave (free): https://wandb.ai\n"
    )
    sys.exit(1)

# ── 2. Weave init — BEFORE OpenAI client is created ──────────────────────────
import weave
from config import WANDB_PROJECT

weave.init(WANDB_PROJECT)

# ── 3. Now safe to import everything that touches llm.py ─────────────────────
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from onboarding import get_or_create_profile, run_questionnaire, save_profile

console = Console()


def _run_turn(graph, query: str, profile: dict, thread_id: str) -> str:
    """Invoke the advisor graph for one user turn and return the final answer."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "query": query,
        "profile": profile,
        "messages": [{"role": "user", "content": query}],
    }
    result = graph.invoke(initial_state, config=config)
    return result.get("final", "(No answer generated.)")


@weave.op(name="run_turn")
def run_turn_traced(graph, query: str, profile: dict, thread_id: str) -> str:
    """Top-level @weave.op wrapping a full turn so the Weave tree is:
    run_turn → router → agents_runner → [agent_*] → synthesizer → fact_check
    """
    return _run_turn(graph, query, profile, thread_id)


def _print_answer(answer: str) -> None:
    console.print()
    console.print(Panel(Markdown(answer), title="[bold cyan]Advisor[/bold cyan]", border_style="cyan"))
    console.print()


def _print_routing(graph, thread_id: str) -> None:
    """Show which agents were called after each turn (debug info)."""
    try:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        route = state.values.get("route", [])
        reason = state.values.get("route_reason", "")
        if route:
            console.print(
                f"[dim]  ↳ Agents called: [bold]{', '.join(route)}[/bold]  |  {reason}[/dim]"
            )
    except Exception:
        pass


def main() -> None:
    # ── Onboarding ────────────────────────────────────────────────────────────
    profile = get_or_create_profile()
    thread_id = f"user_{profile['name'].lower().replace(' ', '_')}"

    # ── Import graph (after weave.init) ───────────────────────────────────────
    from graph import advisor_graph as graph

    # ── Print Weave project URL ───────────────────────────────────────────────
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    if wandb_entity:
        weave_url = f"https://wandb.ai/{wandb_entity}/{WANDB_PROJECT}/weave"
    else:
        weave_url = f"https://wandb.ai/home  (open W&B → {WANDB_PROJECT} → Weave)"
    console.print(f"[dim]📊 Traces: {weave_url}[/dim]")
    console.print(f"[dim]💬 Session thread: [bold]{thread_id}[/bold][/dim]")
    console.print()
    console.print(Rule("[bold cyan]International Student Advisor[/bold cyan]", style="cyan"))
    console.print(
        "[dim]Ask anything about visa rules, academics, finances, jobs, or taxes.\n"
        "Commands: [bold]exit[/bold] | [bold]update profile[/bold][/dim]\n"
    )

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye! Good luck with your studies.[/dim]")
            break

        if query.lower() == "update profile":
            profile = run_questionnaire(profile["name"])
            save_profile(profile)
            console.print("[bold green]Profile updated![/bold green]\n")
            continue

        console.print("[dim]Thinking…[/dim]")
        try:
            answer = run_turn_traced(graph, query, profile, thread_id)
            _print_answer(answer)
            _print_routing(graph, thread_id)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            console.print("[dim]Please try again. If the error persists, check your API keys.[/dim]\n")


if __name__ == "__main__":
    main()
