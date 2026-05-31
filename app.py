"""i-House — International Student Advisor (Streamlit frontend).

UI by anvipoluri (https://github.com/anvipoluri/atlas-of-aid).
Backend integration by AriS10223.

Startup order (mirrors main.py — critical for Weave SDK patching):
  1. load_dotenv()
  2. weave.init()   — BEFORE OpenAI client is created
  3. graph import   — llm.py's get_client() is lazy, so Weave patch is in place
"""
from __future__ import annotations

import base64
import html
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ── 1. Env ────────────────────────────────────────────────────────────────────
load_dotenv()

# ── 2. Weave init — MUST precede any llm.py import ───────────────────────────
import weave
import streamlit as st
from config import WANDB_PROJECT

@st.cache_resource
def _init_weave():
    weave.init(WANDB_PROJECT)

_init_weave()

# ── 3. Backend imports (safe now that Weave has patched the SDK) ──────────────
from onboarding import save_profile
from tools import reset_turn_tracking

@st.cache_resource
def _get_graph():
    from graph import advisor_graph
    return advisor_graph


# ── Hero image — fetch from public repo if not present locally ────────────────
APP_DIR   = Path(__file__).parent
HERO_PATH = APP_DIR / "src" / "assets" / "hero-painting.jpg"
HERO_URL  = "https://raw.githubusercontent.com/anvipoluri/atlas-of-aid/main/src/assets/hero-painting.jpg"

@st.cache_resource
def _ensure_hero() -> bool:
    if HERO_PATH.exists():
        return True
    HERO_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(HERO_URL, HERO_PATH)
        return True
    except Exception:
        return False

_hero_available = _ensure_hero()


# ── Profile field definitions (friend's UI → our backend mapping) ─────────────
PROFILE_FIELDS = [
    ("name",        "What's your name?",         "e.g. John Appleseed"),
    ("school",      "Where do you go to school?", "e.g. Northeastern University"),
    ("nationality", "Nationality?",               "e.g. India"),
    ("visa",        "What visa do you hold?",     "e.g. F-1"),
    ("study",       "What do you study?",         "e.g. B.S. in Computer Science"),
    ("plan",        "What's your plan after graduating?", "e.g. Work in NYC on OPT"),
]


def _map_profile(p: dict) -> dict:
    """Translate the UI's field names to the backend AgentState profile format."""
    return {
        "name":            p.get("name",        "Student"),
        "university":      p.get("school",      "Not specified"),
        "nationality":     p.get("nationality", "Not specified"),
        "visa_status":     p.get("visa",        "Not specified"),
        "field_of_study":  p.get("study",       "Not specified"),
        "post_study_plan": p.get("plan",        "Not specified"),
        "time_in_usa":     "Not specified",
    }


# ── Backend call — replaces the REST API call in the original file ────────────
def run_turn(message: str, profile: dict, thread_id: str) -> dict:
    """Invoke the LangGraph advisor pipeline directly (no FastAPI layer needed)."""
    backend_profile = _map_profile(profile)
    reset_turn_tracking()
    graph = _get_graph()
    result = graph.invoke(
        {
            "query":    message,
            "profile":  backend_profile,
            "messages": [{"role": "user", "content": message}],
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "reply":    result.get("final", "(No answer generated.)"),
        "thread_id": thread_id,
        "route":    result.get("route", []),
    }


# ═════════════════════════════════════════════════════════════════════════════
# UI — everything below is the friend's original design, unchanged
# ═════════════════════════════════════════════════════════════════════════════

def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def init_state() -> None:
    st.session_state.setdefault("page", "landing")
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)


def navigate(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def logo(size: str = "md", extra_class: str = "") -> str:
    return f"""
    <span class="logo logo-{size} {extra_class}">
      <span style="color: var(--crimson)">i</span><span style="color: var(--ink)">-</span><span style="color: var(--varsity)">H</span><span style="color: var(--emerald)">o</span><span style="color: var(--ember)">u</span><span style="color: var(--crimson)">s</span><span style="color: var(--varsity)">e</span>
    </span>
    """


def icon_svg(name: str) -> str:
    icons = {
        "academic": '<path d="M22 10 12 5 2 10l10 5 10-5Z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/>',
        "legal":    '<path d="m16 16 3-8 3 8c-.9 1.3-5.1 1.3-6 0Z"/><path d="m2 16 3-8 3 8c-.9 1.3-5.1 1.3-6 0Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/>',
        "jobs":     '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><path d="M3 13h18"/>',
        "taxes":    '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M8 6h8"/><path d="M8 10h8"/><path d="M8 14h2"/><path d="M14 14h2"/><path d="M8 18h2"/><path d="M14 18h2"/>',
        "anywhere": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2c3 3 3 17 0 20"/><path d="M12 2c-3 3-3 17 0 20"/>',
    }
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{icons[name]}</svg>'


def icon_rail() -> str:
    items = [
        ("Academic",  "academic"),
        ("Visa & Legal", "legal"),
        ("Jobs",      "jobs"),
        ("Taxes",     "taxes"),
        ("Anywhere",  "anywhere"),
    ]
    buttons = "".join(
        f'<span class="icon-roundel icon-{key}" title="{label}">{icon_svg(key)}</span>'
        for label, key in items
    )
    return f'<div class="icon-rail">{buttons}</div>'


def inject_css() -> None:
    st.html(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=IM+Fell+English+SC&family=IM+Fell+English:ital@0;1&family=Cinzel+Decorative:wght@400;700;900&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap");

        :root {
          --radius: 0.5rem;
          --background: oklch(0.955 0.028 82);
          --foreground: oklch(0.22 0.04 50);
          --parchment: oklch(0.93 0.04 80);
          --ink: oklch(0.18 0.03 45);
          --varsity: oklch(0.46 0.17 252);
          --crimson: oklch(0.48 0.19 25);
          --emerald: oklch(0.50 0.13 155);
          --ember: oklch(0.68 0.18 52);
          --gold: oklch(0.78 0.14 85);
          --card: oklch(0.96 0.03 82);
          --muted-foreground: oklch(0.42 0.04 55);
          --gradient-gold: linear-gradient(135deg, oklch(0.85 0.12 85), oklch(0.62 0.14 70), oklch(0.88 0.14 90), oklch(0.55 0.13 65));
          --gradient-parchment: linear-gradient(180deg, oklch(0.96 0.03 82), oklch(0.91 0.045 78));
          --shadow-frame: 0 30px 60px -20px oklch(0.18 0.03 45 / 0.55), 0 8px 20px -8px oklch(0.18 0.03 45 / 0.4);
          --shadow-glow-gold-hover: 0 0 24px 4px oklch(0.78 0.14 85 / 0.55), 0 0 60px 8px oklch(0.68 0.18 52 / 0.25);
          --shadow-sticky: 4px 8px 16px -4px oklch(0.18 0.03 45 / 0.35), 1px 2px 4px oklch(0.18 0.03 45 / 0.2);
          --font-display: "Cinzel Decorative", "IM Fell English SC", "Times New Roman", serif;
          --font-ornate: "IM Fell English", "EB Garamond", "Times New Roman", serif;
          --font-serif: "EB Garamond", "Times New Roman", Georgia, serif;
          --font-header: "Times New Roman", "EB Garamond", Georgia, serif;
        }

        .stApp {
          background-color: var(--background);
          background-image:
            radial-gradient(ellipse at top, oklch(0.92 0.05 75 / 0.7), transparent 60%),
            radial-gradient(ellipse at bottom, oklch(0.88 0.06 60 / 0.5), transparent 70%);
          background-attachment: fixed;
          color: var(--foreground);
          font-family: var(--font-serif);
        }
        .block-container {
          max-width: 1080px;
          padding: 2rem 1rem 5rem;
        }
        #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stDecoration"] {
          display: none;
        }
        h1, h2, h3, p, label, .stMarkdown, .stTextInput label {
          font-family: var(--font-serif);
          color: var(--foreground);
        }

        .logo {
          display: inline-block;
          font-family: var(--font-display);
          font-weight: 900;
          letter-spacing: 0;
          text-shadow: 0 1px 0 oklch(0.98 0.05 88 / 0.6), 0 2px 14px oklch(0.78 0.14 85 / 0.35);
          white-space: nowrap;
        }
        .logo-xl { font-size: clamp(3.5rem, 8vw, 6rem); }
        .logo-md { font-size: clamp(2.2rem, 4vw, 3rem); }
        .logo-sm { font-size: 1.65rem; }

        .hero-wrap {
          min-height: 58vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }
        .hero-card {
          width: min(100%, 980px);
          animation: fade-up 0.7s ease-out both;
        }
        .gold-frame {
          position: relative;
          padding: clamp(14px, 2.2vw, 28px);
          background: var(--gradient-gold);
          background-size: 200% 200%;
          box-shadow: var(--shadow-frame);
          border-radius: 4px;
        }
        .gold-frame::before,
        .gold-frame::after {
          content: "";
          position: absolute;
          inset: 6px;
          border: 1px solid oklch(0.35 0.08 60 / 0.55);
          border-radius: 2px;
          pointer-events: none;
          z-index: 3;
        }
        .gold-frame::after {
          inset: 10px;
          border-color: oklch(0.95 0.08 88 / 0.6);
        }
        .hero-painting {
          position: relative;
          overflow: hidden;
          aspect-ratio: 16 / 9;
        }
        .hero-painting img {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        .hero-painting::after {
          content: "";
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse at center, oklch(0.18 0.03 45 / 0.05) 0%, oklch(0.12 0.03 45 / 0.55) 80%);
        }
        .hero-title {
          position: absolute;
          inset: 0;
          z-index: 2;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 1.5rem;
        }
        .tagline {
          margin-top: 0.75rem;
          font-family: var(--font-ornate);
          font-style: italic;
          font-size: clamp(1.15rem, 2.2vw, 1.35rem);
          color: oklch(0.97 0.04 85);
          animation: fade-up 0.7s ease-out 0.3s both;
        }
        .question {
          margin: 2.5rem 0 1rem;
          text-align: center;
          font-family: var(--font-ornate);
          font-style: italic;
          font-size: 1.35rem;
          color: color-mix(in oklch, var(--foreground), transparent 20%);
        }
        .microcopy {
          margin-top: 0.75rem;
          text-align: center;
          font-family: var(--font-ornate);
          font-size: 0.75rem;
          letter-spacing: 0.25em;
          text-transform: uppercase;
          color: var(--muted-foreground);
        }

        .stButton > button, .stFormSubmitButton > button {
          position: relative;
          min-height: 2.8rem;
          padding: 0.65rem 1.7rem;
          border-radius: 6px;
          border: 1px solid oklch(0.78 0.14 85 / 0.7);
          color: oklch(0.97 0.02 80);
          background: linear-gradient(180deg, oklch(0.55 0.17 252), oklch(0.36 0.15 252));
          box-shadow: inset 0 1px 0 oklch(0.98 0.05 88 / 0.25);
          font-family: var(--font-ornate);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          transition: box-shadow 320ms ease, transform 220ms ease, background 320ms ease;
        }
        .stButton > button:hover, .stButton > button:focus,
        .stFormSubmitButton > button:hover, .stFormSubmitButton > button:focus {
          color: oklch(0.97 0.02 80);
          border-color: oklch(0.78 0.14 85 / 0.8);
          box-shadow: var(--shadow-glow-gold-hover), inset 0 1px 0 oklch(0.98 0.05 88 / 0.35);
          transform: translateY(-1px);
        }

        .top-logo-left {
          position: fixed;
          top: 1.4rem;
          left: 1.5rem;
          z-index: 5;
          animation: logo-shrink 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        }
        .top-logo-center {
          text-align: center;
          margin-bottom: 1.5rem;
          animation: logo-shrink 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        }
        .form-shell {
          max-width: 720px;
          margin: 5rem auto 0;
          animation: unfurl 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
          transform-origin: top center;
        }
        .parchment-panel {
          background: var(--card);
          border-radius: 3px;
          padding: clamp(1.5rem, 4vw, 2.5rem);
        }
        .panel-heading {
          margin: 0;
          text-align: center;
          font-family: var(--font-header);
          font-size: clamp(2rem, 5vw, 2.8rem);
          letter-spacing: 0;
          color: var(--foreground);
        }
        .panel-subtitle {
          margin: 0.2rem 0 1.5rem;
          text-align: center;
          font-family: var(--font-ornate);
          font-style: italic;
          color: var(--muted-foreground);
        }
        div[data-testid="stTextInput"] label {
          font-family: var(--font-ornate);
          font-size: 0.95rem;
          color: color-mix(in oklch, var(--foreground), transparent 20%);
        }
        div[data-testid="stTextInput"] input {
          background: oklch(0.97 0.025 85);
          border: 1px solid oklch(0.55 0.08 60 / 0.45);
          border-bottom: 2px solid oklch(0.45 0.06 55 / 0.55);
          border-radius: 6px;
          color: var(--ink);
          font-family: var(--font-serif);
          font-size: 1rem;
        }
        div[data-testid="stTextInput"] input:focus {
          border-color: oklch(0.78 0.14 85 / 0.9);
          box-shadow: 0 0 0 3px oklch(0.78 0.14 85 / 0.25);
        }
        .privacy-note {
          text-align: center;
          font-size: 0.82rem;
          font-style: italic;
          color: var(--muted-foreground);
          margin-top: 1rem;
        }
        div[data-testid="stForm"] {
          max-width: 720px;
          margin: 1.5rem auto 0;
          padding: clamp(1.2rem, 3vw, 2rem);
          background: var(--card);
          border: 14px solid oklch(0.78 0.14 85 / 0.85);
          border-radius: 4px;
          box-shadow: var(--shadow-frame);
        }

        .icon-rail {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: clamp(0.75rem, 3vw, 1.5rem);
          flex-wrap: wrap;
          margin-top: 2rem;
        }
        .icon-roundel {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 56px;
          height: 56px;
          border-radius: 9999px;
          background: var(--gradient-parchment);
          border: 1px solid oklch(0.78 0.14 85 / 0.7);
          color: var(--ink);
          box-shadow: inset 0 1px 0 oklch(1 0 0 / 0.5), 0 4px 10px -4px oklch(0.18 0.03 45 / 0.35);
          transition: transform 280ms ease, box-shadow 280ms ease, color 280ms ease;
        }
        .icon-roundel:hover {
          transform: translateY(-2px) scale(1.05);
          box-shadow: 0 0 18px 2px oklch(0.78 0.14 85 / 0.55), inset 0 1px 0 oklch(1 0 0 / 0.6);
          color: var(--varsity);
        }
        .icon-roundel svg { display: none; }
        .icon-roundel::before {
          font-family: var(--font-display);
          font-weight: 700;
          font-size: 1.15rem;
          line-height: 1;
        }
        .icon-academic::before { content: "A"; color: var(--varsity); }
        .icon-legal::before    { content: "V"; color: var(--crimson); }
        .icon-jobs::before     { content: "J"; color: var(--emerald); }
        .icon-taxes::before    { content: "T"; color: var(--ember); }
        .icon-anywhere::before { content: "G"; color: var(--varsity); }
        .icon-roundel svg {
          width: 0; height: 0;
          fill: none; stroke: currentColor;
          stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round;
        }

        .welcome-title {
          text-align: center;
          font-family: var(--font-header);
          font-size: clamp(2.6rem, 7vw, 4rem);
          margin: 6vh 0 2rem;
          letter-spacing: 0;
          animation: fade-up 0.7s ease-out both;
        }
        .chat-stack {
          max-width: 720px;
          margin: 0 auto 2rem;
        }
        .sticky-note {
          position: relative;
          margin: 1.45rem 2.75rem 1.45rem 0;
          background: linear-gradient(180deg, oklch(0.93 0.07 95), oklch(0.88 0.08 90));
          color: var(--ink);
          padding: 1.5rem 1.6rem;
          border-radius: 2px;
          box-shadow: var(--shadow-sticky);
          font-family: var(--font-serif);
          line-height: 1.55;
          animation: fade-up 0.7s ease-out both;
        }
        .sticky-note::before {
          content: "";
          position: absolute;
          top: -10px; left: 50%;
          width: 80px; height: 22px;
          transform: translateX(-50%) rotate(-2deg);
          background: oklch(0.55 0.08 60 / 0.35);
          border: 1px solid oklch(0.35 0.06 55 / 0.35);
          border-radius: 1px;
          box-shadow: 0 2px 4px oklch(0.18 0.03 45 / 0.2);
        }
        .sticky-user {
          margin-left: 2.75rem;
          margin-right: 0;
          background: linear-gradient(180deg, oklch(0.85 0.12 50), oklch(0.78 0.13 45));
          color: oklch(0.16 0.03 45);
        }
        .tilt-l { transform: rotate(-1.2deg); }
        .tilt-r { transform: rotate(1.1deg); }
        .note-author {
          margin: 0 0 0.5rem;
          font-family: var(--font-ornate);
          font-size: 0.72rem;
          letter-spacing: 0.2em;
          text-transform: uppercase;
          opacity: 0.7;
        }
        .note-text {
          margin: 0;
          font-size: 1.04rem;
          white-space: pre-wrap;
        }
        .chat-composer { max-width: 720px; margin: 0 auto; }
        .composer-frame {
          margin-top: 1rem;
          animation: fade-up 0.7s ease-out 0.2s both;
        }

        @keyframes fade-up {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes unfurl {
          0%   { opacity: 0; transform: scaleY(0.02) scaleX(0.6); }
          50%  { opacity: 1; transform: scaleY(0.05) scaleX(1); }
          100% { opacity: 1; transform: scaleY(1)    scaleX(1); }
        }
        @keyframes logo-shrink {
          from { transform: scale(2.2) translateY(28vh); opacity: 0; }
          to   { transform: scale(1)   translateY(0);    opacity: 1; }
        }
        @media (max-width: 640px) {
          .block-container { padding-inline: 0.8rem; }
          .sticky-note  { margin-right: 0.6rem; }
          .sticky-user  { margin-left:  0.6rem; }
          .top-logo-left { position: absolute; }
        }
        </style>
        """,
    )


# ── Page renderers ────────────────────────────────────────────────────────────

def render_landing() -> None:
    if _hero_available:
        img_src = image_data_uri(HERO_PATH)
    else:
        img_src = HERO_URL

    st.html(
        f"""
        <main class="hero-wrap">
          <section class="hero-card">
            <div class="gold-frame">
              <div class="hero-painting">
                <img src="{img_src}" alt="An oil painting of a gothic university courtyard">
                <div class="hero-title">
                  {logo("xl")}
                  <p class="tagline">A companion for international students.</p>
                </div>
              </div>
            </div>
            <p class="question">Have you been here before?</p>
          </section>
        </main>
        """,
    )
    left, right = st.columns(2)
    with left:
        if st.button("First time", use_container_width=True):
            navigate("welcome")
    with right:
        if st.button("Welcome back", use_container_width=True):
            navigate("chat" if st.session_state.profile else "welcome")
    st.html(icon_rail())
    st.html('<p class="microcopy">Academic · Visa · Jobs · Taxes</p>')


def render_welcome() -> None:
    profile = st.session_state.profile or {}
    st.html(f'<div class="top-logo-left">{logo("sm")}</div>')
    st.html(
        """
        <section class="form-shell">
          <div class="gold-frame">
            <div class="parchment-panel">
              <h1 class="panel-heading">Nice to meet you!</h1>
              <p class="panel-subtitle">We have a few questions...</p>
        """,
    )
    with st.form("profile_form"):
        values = {}
        for key, label, placeholder in PROFILE_FIELDS:
            values[key] = st.text_input(
                label,
                value=profile.get(key, ""),
                placeholder=placeholder,
                key=f"profile_{key}",
            )
        st.html('<p class="privacy-note">This information is only used to make your results as accurate as possible.</p>')
        submitted = st.form_submit_button("Begin")
    st.html("</div></div></section>")

    if submitted:
        if not values["name"].strip():
            st.warning("Please add your name before beginning.")
            return
        new_profile = {k: v.strip() for k, v in values.items()}
        st.session_state.profile   = new_profile
        st.session_state.messages  = []
        st.session_state.thread_id = f"user_{new_profile['name'].lower().replace(' ', '_')}"
        # Persist to Supabase + Redis
        try:
            save_profile(_map_profile(new_profile))
        except Exception:
            pass
        navigate("chat")


def render_empty_chat(profile: dict) -> None:
    first_name = (profile.get("name", "").split() or [""])[0] or profile.get("name", "")
    st.html(
        f"""
        <div class="top-logo-center">{logo("md")}</div>
        <h1 class="welcome-title">Welcome, <span style="color: var(--crimson)">{html.escape(first_name)}</span>!</h1>
        """,
    )
    render_chat_form("Ask")
    st.html(icon_rail())
    st.html('<p class="microcopy">Academic · Visa · Jobs · Taxes</p>')


def render_messages(profile: dict) -> None:
    first_name = (profile.get("name", "").split() or [""])[0] or profile.get("name", "You")
    notes = []
    for index, message in enumerate(st.session_state.messages):
        role   = message["role"]
        author = first_name if role == "user" else "i-House"
        classes = ["sticky-note", "tilt-l" if index % 2 == 0 else "tilt-r"]
        if role == "user":
            classes.append("sticky-user")
        notes.append(
            f"""
            <article class="{' '.join(classes)}">
              <p class="note-author">{html.escape(author)}</p>
              <p class="note-text">{html.escape(message["text"])}</p>
            </article>
            """
        )
    st.html(
        f"""
        <div class="top-logo-center">{logo("md")}</div>
        <section class="chat-stack">{''.join(notes)}</section>
        """,
    )
    render_chat_form("Send")


def render_chat_form(button_label: str) -> None:
    with st.form(f"chat_form_{len(st.session_state.messages)}", clear_on_submit=True):
        text = st.text_input("What's up?", label_visibility="collapsed", placeholder="What's up?")
        submitted = st.form_submit_button(button_label)

    if submitted and text.strip():
        user_text = text.strip()
        st.session_state.messages.append(
            {"id": str(uuid.uuid4()), "role": "user", "text": user_text}
        )
        with st.spinner("Consulting the archives..."):
            try:
                response = run_turn(
                    user_text,
                    st.session_state.profile,
                    st.session_state.thread_id,
                )
                reply = response.get("reply", "I could not generate a reply.")
            except Exception as exc:
                reply = f"Something went wrong: {exc}"
        st.session_state.messages.append(
            {"id": str(uuid.uuid4()), "role": "assistant", "text": reply}
        )
        st.rerun()


def render_chat() -> None:
    if not st.session_state.profile:
        navigate("welcome")
    if st.session_state.messages:
        render_messages(st.session_state.profile)
    else:
        render_empty_chat(st.session_state.profile)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="i-House — A companion for international students",
        page_icon="i-House",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_state()
    inject_css()

    page = st.session_state.page
    if page == "welcome":
        render_welcome()
    elif page == "chat":
        render_chat()
    else:
        render_landing()


if __name__ == "__main__":
    main()
