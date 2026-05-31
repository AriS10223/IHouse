"""First-session onboarding and profile management.

Profiles are stored locally in profiles.json keyed by (lowercased) name.
On first run the user answers 7 questions; subsequent runs greet them and
skip the questionnaire.  An 'update profile' command in main.py calls
run_questionnaire() again to refresh answers.
"""
from __future__ import annotations

import os
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

import cache

console = Console()

# ── Questionnaire definition ─────────────────────────────────────────────────

QUESTIONS: list[dict] = [
    {
        "key": "university",
        "prompt": "Which university do you attend?",
        "example": "e.g. University of Illinois Urbana-Champaign",
    },
    {
        "key": "nationality",
        "prompt": "What is your nationality (country of citizenship)?",
        "example": "e.g. India",
    },
    {
        "key": "visa_status",
        "prompt": "What is your current visa status?",
        "example": "e.g. F-1, J-1, OPT, CPT, on a different visa",
    },
    {
        "key": "field_of_study",
        "prompt": "What is your field of study / major?",
        "example": "e.g. Computer Science, MBA, Mechanical Engineering",
    },
    {
        "key": "post_study_plan",
        "prompt": "What is your plan after your studies?",
        "example": "Get a job (OPT → H-1B) | Do internships and return home | Unsure",
    },
    {
        "key": "time_in_usa",
        "prompt": "How long have you been in the USA? (Be specific — it matters for taxes.)",
        "example": "e.g. Just arrived, 3 months, 1 year, 4 years",
    },
]


# ── Profile I/O ───────────────────────────────────────────────────────────────

def load_profile(name: str) -> Optional[dict]:
    """Return the saved profile for *name* (case-insensitive), or None."""
    return cache.get_profile(name)


def save_profile(profile: dict) -> None:
    """Persist (or overwrite) a profile — writes Supabase + Redis."""
    cache.set_profile(profile)


# ── Questionnaire runner ──────────────────────────────────────────────────────

def run_questionnaire(name: str) -> dict:
    """Ask the onboarding questions and return a completed profile dict."""
    console.print(
        Panel(
            "[bold cyan]Welcome! Let me learn a little about you so I can give "
            "you personalised advice.[/bold cyan]\n"
            "[dim](Your answers are saved locally and never sent anywhere except "
            "the advisor AI.)[/dim]",
            title="[bold]First-Time Setup[/bold]",
            border_style="cyan",
        )
    )
    profile: dict = {"name": name.strip()}
    for q in QUESTIONS:
        console.print(f"\n[yellow]{q['prompt']}[/yellow]")
        console.print(f"[dim]{q['example']}[/dim]")
        answer = Prompt.ask("[bold]Your answer[/bold]").strip()
        profile[q["key"]] = answer or "Not specified"
    return profile


# ── Entry point called by main.py ─────────────────────────────────────────────

def get_or_create_profile() -> dict:
    """Ask for the user's name, load or create their profile, and return it."""
    console.print(
        Panel(
            "[bold white]International Student Advisor[/bold white]\n"
            "[dim]Powered by W&B Weave · LangGraph · OpenRouter[/dim]",
            border_style="bright_blue",
        )
    )
    name = Prompt.ask("\n[bold cyan]What's your name?[/bold cyan]").strip()
    if not name:
        name = "Student"

    profile = load_profile(name)
    if profile:
        console.print(
            f"\n[bold green]Welcome back, {profile['name']}![/bold green]  "
            f"([dim]{profile['university']} · {profile['visa_status']}[/dim])\n"
            f"[dim]Type [bold]update profile[/bold] at any time to edit your details.[/dim]\n"
        )
    else:
        profile = run_questionnaire(name)
        save_profile(profile)
        console.print(
            f"\n[bold green]Profile saved![/bold green]  "
            f"Welcome, {profile['name']}!\n"
        )
    return profile


# ── Prompt formatter ──────────────────────────────────────────────────────────

def format_profile_for_prompt(profile: dict) -> str:
    """Render the profile as a compact context block for LLM system prompts."""
    return (
        f"STUDENT PROFILE:\n"
        f"  Name:            {profile.get('name', 'Unknown')}\n"
        f"  University:      {profile.get('university', 'Unknown')}\n"
        f"  Nationality:     {profile.get('nationality', 'Unknown')}\n"
        f"  Visa status:     {profile.get('visa_status', 'Unknown')}\n"
        f"  Field of study:  {profile.get('field_of_study', 'Unknown')}\n"
        f"  Post-study plan: {profile.get('post_study_plan', 'Unknown')}\n"
        f"  Time in USA:     {profile.get('time_in_usa', 'Unknown')}\n"
    )
