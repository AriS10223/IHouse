"""Streamlit frontend for the International Student Advisor (i-House).

Startup order (mirrors main.py — critical for Weave patching):
  1. load_dotenv()
  2. weave.init()   — BEFORE OpenAI client is created
  3. graph import   — llm.py's get_client() is lazy, so patching is already in place

Pages (session_state.page):
  landing    -> onboarding (first time) | chat (returning user)
  onboarding -> chat
  chat       -> persistent chat with real graph backend
"""
from __future__ import annotations

import os
import time
from dotenv import load_dotenv

# ── 1. Environment ────────────────────────────────────────────────────────────
load_dotenv()

# ── 2. Weave init (must precede any llm.py usage) ────────────────────────────
import weave
import streamlit as st
from config import WANDB_PROJECT

@st.cache_resource
def _init_weave():
    weave.init(WANDB_PROJECT)

_init_weave()

# ── 3. Backend imports (safe now that Weave has patched the SDK) ──────────────
from onboarding import save_profile, load_profile, QUESTIONS
from tools import reset_turn_tracking, get_turn_searches

@st.cache_resource
def _get_graph():
    from graph import advisor_graph
    return advisor_graph


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="i-House — International Student Advisor",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ── Design system CSS (academia / parchment theme) ────────────────────────────
def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=IM+Fell+English:ital@0;1&display=swap');

        /* ── Force light / parchment colour scheme regardless of OS dark mode ── */
        :root, html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"], .stApp {
            color-scheme: light !important;
        }

        /* ── Design tokens (explicit hex — not var() where Streamlit can clobber) ── */
        :root {
            --parchment:    #F7F0E3;
            --parchment-dk: #EDE3CF;
            --parchment-md: #E4D9C0;
            --ink:          #2C2416;
            --ink-soft:     #5A4A2A;
            --crimson:      #8B1A1A;
            --crimson-dk:   #6B1010;
            --crimson-lt:   #A83030;
            --gold:         #C9A84C;
            --gold-lt:      #E5C97A;
            --blue:         #1E3A6E;
            --blue-lt:      #2E5198;
            --shadow-sm:    rgba(44,36,22,0.12);
            --shadow-md:    rgba(44,36,22,0.22);
        }

        /* ── Root surfaces — override Streamlit dark injection ── */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        section[data-testid="stSidebar"] {
            background-color: #F7F0E3 !important;
            color: #2C2416 !important;
        }
        .block-container {
            padding-top: 2.5rem !important;
            max-width: 820px !important;
            background-color: #F7F0E3 !important;
        }

        /* ── Typography — explicit colours, never inherit from Streamlit dark ── */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Cinzel Decorative', 'Times New Roman', serif !important;
            color: #8B1A1A !important;
        }
        p, li, span, label, div,
        .stMarkdown p, .stMarkdown li, .stMarkdown span,
        [data-testid="stMarkdownContainer"] * {
            color: #2C2416 !important;
            font-family: 'EB Garamond', Georgia, serif !important;
            font-size: 1.05rem !important;
            line-height: 1.65 !important;
        }

        /* ── Gold frame ── */
        .gold-frame {
            border: 2px solid #C9A84C;
            box-shadow: 0 0 0 1px #EDE3CF, 0 0 0 4px #C9A84C, 0 6px 24px rgba(44,36,22,0.15);
            border-radius: 10px;
            padding: 2rem 2.5rem;
            background: #F7F0E3 !important;
            color: #2C2416 !important;
            margin-bottom: 1.5rem;
        }

        /* ═══════════════════════════════════════════════════════
           BUTTONS — rounded, tactile, distinct states
           Radius 14px = friendly rounded, not full-pill
           Box-shadow gives depth; transform gives press feel
        ═══════════════════════════════════════════════════════ */
        .stButton > button,
        .stFormSubmitButton > button {
            font-family: 'IM Fell English', Georgia, serif !important;
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            border-radius: 14px !important;
            padding: 0.6rem 1.8rem !important;
            min-height: 46px !important;
            cursor: pointer !important;
            transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease !important;
            letter-spacing: 0.02em !important;
        }

        /* ── Primary / crimson CTA ── */
        .btn-crimson > button,
        .btn-crimson .stButton > button {
            background: linear-gradient(145deg, #A32020, #8B1A1A) !important;
            color: #FFF8EE !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(139,26,26,0.40), 0 1px 3px rgba(44,36,22,0.20) !important;
        }
        .btn-crimson > button:hover,
        .btn-crimson .stButton > button:hover {
            background: linear-gradient(145deg, #B83030, #A32020) !important;
            box-shadow: 0 6px 20px rgba(139,26,26,0.55), 0 2px 6px rgba(44,36,22,0.20) !important;
            transform: translateY(-2px) !important;
        }
        .btn-crimson > button:active,
        .btn-crimson .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 8px rgba(139,26,26,0.35) !important;
        }

        /* ── Secondary / blue CTA ── */
        .btn-blue > button,
        .btn-blue .stButton > button {
            background: linear-gradient(145deg, #2E5198, #1E3A6E) !important;
            color: #EEF2FF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(30,58,110,0.40), 0 1px 3px rgba(44,36,22,0.20) !important;
        }
        .btn-blue > button:hover,
        .btn-blue .stButton > button:hover {
            background: linear-gradient(145deg, #3A62B0, #2E5198) !important;
            box-shadow: 0 6px 20px rgba(30,58,110,0.55), 0 2px 6px rgba(44,36,22,0.20) !important;
            transform: translateY(-2px) !important;
        }
        .btn-blue > button:active,
        .btn-blue .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 8px rgba(30,58,110,0.35) !important;
        }

        /* ── Gold / submit buttons (forms) ── */
        .stFormSubmitButton > button {
            background: linear-gradient(145deg, #D4B355, #C9A84C) !important;
            color: #2C2416 !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(201,168,76,0.40), 0 1px 3px rgba(44,36,22,0.15) !important;
        }
        .stFormSubmitButton > button:hover {
            background: linear-gradient(145deg, #E5C97A, #D4B355) !important;
            box-shadow: 0 6px 20px rgba(201,168,76,0.55) !important;
            transform: translateY(-2px) !important;
        }
        .stFormSubmitButton > button:active {
            transform: translateY(0px) !important;
        }

        /* ── Sidebar buttons ── */
        section[data-testid="stSidebar"] .stButton > button {
            background: #EDE3CF !important;
            color: #2C2416 !important;
            border: 1.5px solid #C9A84C !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 6px rgba(44,36,22,0.12) !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #E4D9C0 !important;
            box-shadow: 0 4px 10px rgba(44,36,22,0.18) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Inputs — parchment background, gold focus ── */
        .stTextInput > div > div > input,
        .stTextArea textarea {
            background: #EDE3CF !important;
            border: 1.5px solid #C9A84C !important;
            color: #2C2416 !important;
            font-family: 'EB Garamond', serif !important;
            font-size: 1rem !important;
            border-radius: 10px !important;
            caret-color: #8B1A1A !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: #8B1A1A !important;
            box-shadow: 0 0 0 3px rgba(139,26,26,0.18) !important;
            outline: none !important;
        }
        .stTextInput label, .stTextArea label {
            color: #5A4A2A !important;
            font-family: 'IM Fell English', serif !important;
            font-size: 0.92rem !important;
        }

        /* ── Chat bubbles ── */
        .msg-user {
            background: #FFF9EE !important;
            border-left: 4px solid #8B1A1A;
            border-radius: 4px 12px 12px 4px;
            padding: 0.85rem 1.1rem;
            margin: 0.6rem 0 0.6rem 2.5rem;
            box-shadow: 2px 3px 10px rgba(44,36,22,0.12);
            color: #2C2416 !important;
        }
        .msg-assistant {
            background: #EEF2FA !important;
            border-left: 4px solid #1E3A6E;
            border-radius: 4px 12px 12px 4px;
            padding: 0.85rem 1.1rem;
            margin: 0.6rem 2.5rem 0.6rem 0;
            box-shadow: 2px 3px 10px rgba(44,36,22,0.12);
            color: #2C2416 !important;
        }
        .msg-label-user {
            font-size: 0.76rem !important;
            color: #8B1A1A !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            margin-bottom: 0.3rem !important;
        }
        .msg-label-assistant {
            font-size: 0.76rem !important;
            color: #1E3A6E !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            margin-bottom: 0.3rem !important;
        }
        .msg-route {
            font-size: 0.75rem !important;
            color: #7A6A4A !important;
            font-style: italic !important;
            margin-top: 0.4rem !important;
            padding-top: 0.35rem !important;
            border-top: 1px solid #D9CDB0 !important;
        }

        /* ── Icon rail pills ── */
        .icon-rail {
            display: flex;
            gap: 0.75rem;
            justify-content: center;
            flex-wrap: wrap;
            margin: 1rem 0;
        }
        .icon-pill {
            background: #EDE3CF !important;
            border: 1.5px solid #C9A84C !important;
            border-radius: 20px !important;
            padding: 0.3rem 1rem !important;
            font-family: 'IM Fell English', serif !important;
            font-size: 0.88rem !important;
            color: #2C2416 !important;
            white-space: nowrap !important;
        }

        /* ── Sidebar content ── */
        section[data-testid="stSidebar"] * {
            color: #2C2416 !important;
            background-color: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
            font-family: 'EB Garamond', serif !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #C9A84C !important;
            opacity: 0.5 !important;
        }

        /* ── Spinner text ── */
        [data-testid="stSpinner"] * { color: #8B1A1A !important; }

        /* ── Error / warning boxes ── */
        [data-testid="stAlert"] {
            background: #FFF3F0 !important;
            border-left-color: #8B1A1A !important;
            color: #2C2416 !important;
            border-radius: 10px !important;
        }

        /* ── Divider ── */
        hr { border-color: #C9A84C !important; opacity: 0.35 !important; }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden !important; }
        [data-testid="stDecoration"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _icon_rail():
    st.markdown(
        """
        <div class="icon-rail">
          <span class="icon-pill">&#127891; Academic</span>
          <span class="icon-pill">&#9878; Visa &amp; Legal</span>
          <span class="icon-pill">&#128188; Jobs</span>
          <span class="icon-pill">&#128203; Taxes</span>
          <span class="icon-pill">&#127760; Anywhere</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_turn(query: str, profile: dict, thread_id: str) -> tuple[str, list[str], str]:
    """Call the advisor graph; return (answer, route, route_reason)."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "query": query,
            "profile": profile,
            "messages": [{"role": "user", "content": query}],
        },
        config=config,
    )
    answer = result.get("final", "(No answer generated.)")
    route = result.get("route", [])
    reason = result.get("route_reason", "")
    return answer, route, reason


# ── Page: Landing ─────────────────────────────────────────────────────────────

def page_landing():
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <h1 style="font-size:2.6rem; margin-bottom:0.2rem;">i-House</h1>
            <p style="font-family:'IM Fell English',serif; font-size:1.2rem; color:#5A4A2A; margin-top:0;">
                A companion for international students
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gold-frame" style="text-align:center;">
            <p style="font-family:'EB Garamond',serif; font-size:1.15rem; color:#3A2E1A; margin:0 0 1rem;">
                Warm, plain-spoken guidance on U.S. visa rules, taxes, jobs,<br>and academic life — tailored to your situation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _icon_rail()
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown('<div class="btn-crimson">', unsafe_allow_html=True)
        if st.button("First time here", use_container_width=True):
            st.session_state.page = "onboarding"
            st.session_state.returning = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="btn-blue">', unsafe_allow_html=True)
        if st.button("Welcome back", use_container_width=True):
            st.session_state.page = "onboarding"
            st.session_state.returning = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.85rem; margin-top:2rem;'>"
        "Powered by LangGraph &middot; W&amp;B Weave &middot; Groq LPU"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Page: Onboarding ──────────────────────────────────────────────────────────

def page_onboarding():
    returning = st.session_state.get("returning", False)

    st.markdown(
        "<h2 style='text-align:center; font-size:1.8rem;'>Nice to meet you</h2>"
        "<p style='text-align:center; font-family:\"IM Fell English\",serif; color:#5A4A2A;'>"
        "Tell i-House a little about yourself so answers fit your situation.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='gold-frame'>", unsafe_allow_html=True)

    with st.form("profile_form"):
        name = st.text_input("Your name *", placeholder="e.g. Priya Sharma")

        if returning:
            # Try to load existing profile on submit — show minimal form
            submitted = st.form_submit_button("Load my profile", use_container_width=True)
            if submitted:
                if not name.strip():
                    st.error("Please enter your name.")
                else:
                    existing = load_profile(name.strip())
                    if existing:
                        st.session_state.profile = existing
                        st.session_state.thread_id = f"user_{name.strip().lower().replace(' ', '_')}"
                        st.session_state.messages = []
                        st.session_state.page = "chat"
                        st.rerun()
                    else:
                        st.warning(f"No profile found for '{name}'. Please fill in your details below.")
                        st.session_state.returning = False
                        st.rerun()
        else:
            university = st.text_input("University", placeholder="e.g. University of Illinois Urbana-Champaign")
            nationality = st.text_input("Nationality", placeholder="e.g. India")
            visa_status = st.text_input("Visa status", placeholder="e.g. F-1, J-1, OPT, CPT")
            field_of_study = st.text_input("Field of study", placeholder="e.g. Computer Science, MBA")
            post_study_plan = st.text_input("Post-study plan", placeholder="e.g. OPT then H-1B, return home")
            time_in_usa = st.text_input("Time in the USA", placeholder="e.g. Just arrived, 1 year, 3 years")

            submitted = st.form_submit_button("Begin my journey", use_container_width=True)
            if submitted:
                if not name.strip():
                    st.error("Name is required.")
                else:
                    profile = {
                        "name": name.strip(),
                        "university": university.strip() or "Not specified",
                        "nationality": nationality.strip() or "Not specified",
                        "visa_status": visa_status.strip() or "Not specified",
                        "field_of_study": field_of_study.strip() or "Not specified",
                        "post_study_plan": post_study_plan.strip() or "Not specified",
                        "time_in_usa": time_in_usa.strip() or "Not specified",
                    }
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.session_state.thread_id = f"user_{name.strip().lower().replace(' ', '_')}"
                    st.session_state.messages = []
                    st.session_state.page = "chat"
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back to home"):
        st.session_state.page = "landing"
        st.rerun()


# ── Page: Chat ────────────────────────────────────────────────────────────────

def page_chat():
    profile = st.session_state.get("profile", {})
    messages = st.session_state.get("messages", [])
    thread_id = st.session_state.get("thread_id", "default")
    first_name = profile.get("name", "").split()[0] if profile.get("name") else "there"

    # Header
    st.markdown(
        f"<h2 style='text-align:center; font-size:1.7rem;'>Hello, "
        f"<span style='color:var(--crimson)'>{first_name}</span></h2>"
        f"<p style='text-align:center; font-family:\"IM Fell English\",serif; color:#5A4A2A; margin-top:-0.5rem;'>"
        f"Ask me anything about visa rules, academics, finances, jobs, or taxes.</p>",
        unsafe_allow_html=True,
    )

    _icon_rail()
    st.markdown("<hr>", unsafe_allow_html=True)

    # Chat history
    if not messages:
        st.markdown(
            "<p style='text-align:center; color:#999; font-style:italic; margin:2rem 0;'>"
            "Your conversation will appear here.</p>",
            unsafe_allow_html=True,
        )
    else:
        for msg in messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='msg-user'>"
                    f"<div class='msg-label-user'>You</div>"
                    f"{msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                route_html = ""
                if msg.get("route"):
                    agents = ", ".join(msg["route"])
                    route_html = f"<div class='msg-route'>Consulted: {agents}</div>"
                # Render markdown in the assistant response
                import re
                content_escaped = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f"<div class='msg-assistant'>"
                    f"<div class='msg-label-assistant'>i-House Advisor</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                if route_html:
                    st.markdown(route_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Input form
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Ask a question",
                placeholder="e.g. Can I work on campus with an F-1 visa?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and query.strip():
        with st.spinner("Consulting the archives..."):
            reset_turn_tracking()
            t0 = time.time()
            try:
                answer, route, reason = _run_turn(query.strip(), profile, thread_id)
                latency = round(time.time() - t0, 1)
            except Exception as exc:
                answer = f"Something went wrong: {exc}\n\nPlease try again."
                route, reason = [], ""
                latency = 0

        st.session_state.messages.append({"role": "user", "content": query.strip()})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "reason": reason,
            "latency": latency,
        })
        st.rerun()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Profile")
        if profile:
            st.write(f"**Name:** {profile.get('name', '')}")
            st.write(f"**University:** {profile.get('university', '')}")
            st.write(f"**Visa:** {profile.get('visa_status', '')}")
            st.write(f"**Field:** {profile.get('field_of_study', '')}")
        st.markdown("---")
        if st.button("Update profile"):
            st.session_state.page = "onboarding"
            st.session_state.returning = False
            st.rerun()
        if st.button("New session"):
            for key in ("profile", "messages", "thread_id", "page", "returning"):
                st.session_state.pop(key, None)
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

def main():
    _inject_css()

    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    page = st.session_state.page
    if page == "landing":
        page_landing()
    elif page == "onboarding":
        page_onboarding()
    elif page == "chat":
        page_chat()


if __name__ == "__main__":
    main()
