"""
StudyVector — Offline Study Assistant
Ask questions across all your course materials. Works offline. Nothing leaves your machine.

Run (must use project venv — client needs Python 3.10+):
    .venv/bin/python -m streamlit run app.py
"""

import html
import json
import sys
from pathlib import Path

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

_ACTIAN_WHEEL_URL = (
    "https://raw.githubusercontent.com/hackmamba-io/actian-vectorAI-db-beta/main/"
    "actian_vectorai-0.1.0b2-py3-none-any.whl"
)

try:
    from actian_vectorai import VectorAIClient, Field, FilterBuilder
except ImportError as e:
    st.error(
        f"**actian-vectorai** is not available for the Python running this app.\n\n"
        f"- **Interpreter:** `{sys.executable}`\n"
        f"- **Fix:** from the project folder run  \n"
        f"  `.venv/bin/python -m streamlit run app.py`  \n"
        f"  (create venv with **Python 3.10+** if needed, then `pip install -r requirements.txt`)\n\n"
        f"- **Or install the wheel into *this* Python:**  \n"
        f"  `pip install \"{_ACTIAN_WHEEL_URL}\"`"
    )
    st.caption(f"Import error: {e}")
    st.stop()

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION  = "studyvector"
EMBED_MODEL = "all-MiniLM-L6-v2"
SERVER      = "localhost:50051"
TOP_K       = 6
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

MODES = {
    "chat":       "Study Chat",
    "exam":       "Exam Prep",
    "flashcards": "Flashcards",
    "duck":       "Rubber Duck",
}
MODE_ICONS = {"chat": "💬", "exam": "📝", "flashcards": "🗂", "duck": "🦆"}

# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Connecting to VectorAI DB...")
def get_client():
    try:
        c = VectorAIClient(SERVER)
        c.connect()
        c.health_check()
        return c
    except Exception:
        return None


def embed(text: str):
    return load_model().encode([text])[0].tolist()


# ── Search ───────────────────────────────────────────────────────────────────

def search(query: str, course_filter: str | None, top_k: int = TOP_K):
    client = get_client()
    if client is None:
        return []
    vec = embed(query)
    kwargs = {"limit": top_k}
    if course_filter and course_filter != "All courses":
        f = FilterBuilder().must(Field("course").eq(course_filter)).build()
        kwargs["filter"] = f
    try:
        return client.points.search(COLLECTION, vector=vec, **kwargs)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


@st.cache_data(ttl=60, show_spinner=False)
def get_courses() -> list[str]:
    client = get_client()
    if client is None:
        return []
    try:
        results = client.points.search(COLLECTION, vector=[0.0] * 384, limit=200)
        return sorted({r.payload.get("course", "General") for r in results if r.payload})
    except Exception:
        return []


# ── Ollama LLM ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def ollama_available() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=1).status_code == 200
    except Exception:
        return False


def ask_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(Ollama error: {e})"


def stream_ollama(prompt: str):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=120,
            stream=True,
        )
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except Exception as e:
        yield f"\n\n(Ollama error: {e})"


def build_rag_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[{r.payload.get('source','?')} p.{r.payload.get('page','?')}]\n{r.payload.get('text','')}"
        for r in chunks
    )
    return f"""You are a helpful study assistant. Answer the student's question using ONLY the course material excerpts below. Be concise. Cite sources by filename and page number.

Course material:
{context}

Student question: {question}

Answer:"""


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StudyVector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --yo-yellow: #D6FF34;
        --bg: #0a0a0b;
        --bg-mid: #12120f;
        --surface-0: #151613;
        --surface-1: #1D1E19;
        --surface-2: #262722;
        --surface-3: #292A2D;
        --text: #EDEDED;
        --muted: rgba(255,255,255,0.6);
        --card-blue: #7DA2FF;
        --card-mint: #5DFFC0;
        --card-cyan: #71F6FF;
        --card-lavender: #C1ADFF;
        --glow: rgba(214, 255, 52, 0.08);
    }

    /* ═══ RESET ═══════════════════════════════════════════ */
    *, *::before, *::after { box-sizing: border-box; }
    html { -webkit-text-size-adjust: 100%; }
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ═══ DARK BASE ═══════════════════════════════════════ */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .stApp > header { background: transparent !important; }
    .stApp { overflow-x: hidden !important; }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] .main,
    section.main > div { min-width: 0 !important; }

    /* ═══ MAIN CONTAINER ═════════════════════════════════ */
    .block-container {
        max-width: min(780px, 100%) !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 1.5rem 1.25rem 6rem !important;
        min-width: 0 !important;
    }

    /* ═══ TEXT ════════════════════════════════════════════ */
    p, li, span, div { color: var(--text); }
    [data-testid="stAppViewContainer"] h1 {
        font-size: 26px !important; line-height: 34px !important;
        font-weight: 700 !important; color: var(--text) !important;
    }
    [data-testid="stAppViewContainer"] h2 {
        font-size: 20px !important; line-height: 26px !important;
        font-weight: 700 !important; color: var(--text) !important;
    }
    [data-testid="stAppViewContainer"] h3 {
        font-size: 16px !important; line-height: 22px !important;
        font-weight: 700 !important; color: var(--text) !important;
    }
    .stCaption, .stCaption p {
        color: var(--muted) !important;
        font-size: 12px !important;
    }

    /* ═══ SIDEBAR ════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: var(--surface-0) !important;
        border-right: 1px solid var(--surface-3) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] span {
        color: #000 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--surface-3) !important;
        margin: 10px 0 !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 16px !important; line-height: 22px !important;
        font-weight: 700 !important; text-transform: uppercase !important;
        letter-spacing: 1.2px !important; margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 14px !important; line-height: 20px !important;
        font-weight: 700 !important; text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stButton button {
        padding: 9px 16px !important;
        font-size: 13px !important; font-weight: 600 !important;
        min-height: unset !important; border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
        background: var(--yo-yellow) !important;
        color: #000 !important; border: none !important;
    }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background: var(--surface-2) !important;
        color: var(--text) !important; border: none !important;
    }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background: var(--surface-3) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: var(--surface-2); border-radius: 8px; padding: 10px 12px;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: var(--yo-yellow) !important;
        font-size: 22px !important; font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 11px !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: 1.2px !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--surface-2) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 8px !important; font-size: 13px !important;
    }
    /* sr-only for sidebar labels */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        position: absolute !important;
        width: 1px !important; height: 1px !important;
        padding: 0 !important; margin: -1px !important;
        overflow: hidden !important; clip: rect(0,0,0,0) !important;
        white-space: nowrap !important; border: 0 !important;
    }

    /* ═══ BUTTONS (main area) ════════════════════════════ */
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] * {
        background: var(--yo-yellow) !important;
        color: #000 !important; border: none !important;
        border-radius: 9999px !important;
        font-size: 13px !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: 0.96px !important;
    }
    button[data-testid="stBaseButton-primary"] {
        padding: 12px 24px !important;
    }
    button[data-testid="stBaseButton-primary"]:hover { opacity: 0.85 !important; }
    button[data-testid="stBaseButton-secondary"] {
        background: var(--surface-1) !important;
        color: var(--text) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 9999px !important;
        padding: 10px 20px !important;
        font-size: 13px !important; font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: var(--surface-2) !important;
        border-color: var(--yo-yellow) !important;
    }

    /* ═══ CHAT MESSAGES — clean, minimal bubbles ═════════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0.4rem 0 !important;
        margin-bottom: 0 !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    /* Avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        flex-shrink: 0 !important;
        width: 28px !important; height: 28px !important;
        font-size: 16px !important;
    }

    /* Message content — full width, clean look */
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        min-width: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        overflow-wrap: anywhere;
        word-break: break-word;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.5rem 0 !important;
        box-shadow: none !important;
    }

    /* User messages — subtle highlight */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
        background: rgba(214,255,52,0.06) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        border-left: 3px solid rgba(214,255,52,0.4) !important;
    }

    /* Assistant messages — clean with left accent */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stChatMessageContent"] {
        padding: 0.5rem 0 0.5rem 1rem !important;
        border-left: 3px solid var(--surface-3) !important;
        border-radius: 0 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] li {
        color: var(--text) !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
        margin: 0 0 0.4rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0 !important; }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] pre {
        background: var(--surface-1) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        font-size: 13px; line-height: 1.5;
        overflow-x: auto;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] pre code {
        color: var(--text) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p code,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] li code {
        color: var(--text) !important;
        background: var(--surface-2) !important;
        padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] a {
        color: var(--card-blue) !important;
        text-decoration: underline; text-underline-offset: 2px;
    }
    [data-testid="stChatMessage"] .stExpander {
        background: var(--surface-1) !important;
        border: 1px solid var(--surface-2) !important;
        border-radius: 10px; margin-top: 0.5rem;
    }
    [data-testid="stChatMessage"] .stExpander summary { padding: 0.35rem 0 !important; }
    [data-testid="stChatMessage"] .stExpander p,
    [data-testid="stChatMessage"] .stExpander [data-testid="stCaptionContainer"] p {
        font-size: 12px !important; line-height: 1.45 !important;
    }

    /* ═══ CHAT INPUT — kill every wrapper background ════ */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > *,
    [data-testid="stBottom"] > * > *,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > *,
    [data-testid="stChatInput"] > * > *,
    [data-testid="stChatInput"] form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stBottom"] {
        background: transparent !important;
        padding: 0 !important;
    }
    /* The actual visible input bar */
    [data-testid="stChatInputContainer"] {
        background: var(--surface-1) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"]:focus-within [data-testid="stChatInputContainer"] {
        border-color: rgba(214,255,52,0.5) !important;
        box-shadow: 0 0 0 1px rgba(214,255,52,0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text) !important;
        border: none !important;
        font-size: 15px !important;
        line-height: 1.45 !important;
        min-height: 2.5rem !important;
        max-height: 8rem;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
        caret-color: var(--yo-yellow) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
    }
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {
        background: var(--yo-yellow) !important;
        color: #000 !important;
        border-radius: 9999px !important;
        border: none !important;
        min-width: 2.2rem; min-height: 2.2rem;
        margin: 0.2rem 0.25rem 0.2rem 0 !important;
    }

    /* ═══ TEXT INPUTS & TEXTAREAS (non-chat) ═════════════ */
    .stTextInput input, .stTextArea textarea {
        background: var(--surface-1) !important;
        color: var(--text) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        caret-color: var(--yo-yellow) !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: var(--muted) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--yo-yellow) !important;
        box-shadow: 0 0 0 1px var(--yo-yellow) !important;
    }
    /* sr-only for input labels */
    .stTextInput[data-baseweb] label[data-testid="stWidgetLabel"],
    .stTextArea[data-baseweb] label[data-testid="stWidgetLabel"] {
        position: absolute !important;
        width: 1px !important; height: 1px !important;
        padding: 0 !important; margin: -1px !important;
        overflow: hidden !important; clip: rect(0,0,0,0) !important;
        white-space: nowrap !important; border: 0 !important;
    }

    /* ═══ SELECTBOX ══════════════════════════════════════ */
    .stSelectbox > div > div {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 12px !important;
    }
    .stSelectbox svg { fill: var(--muted) !important; }
    [data-baseweb="popover"], [data-baseweb="menu"],
    ul[role="listbox"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--surface-3) !important;
    }
    ul[role="listbox"] li { color: var(--text) !important; }
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {
        background: var(--surface-3) !important;
    }

    /* ═══ SLIDER ═════════════════════════════════════════ */
    .stSlider [data-testid="stThumbValue"] { color: var(--yo-yellow) !important; }
    .stSlider [role="slider"] { background: var(--yo-yellow) !important; }
    .stSlider label { color: var(--muted) !important; font-size: 13px !important; }

    /* ═══ EXPANDERS ══════════════════════════════════════ */
    .stExpander {
        background: var(--surface-1) !important;
        border: 1px solid var(--surface-2) !important;
        border-radius: 12px !important;
        margin-top: 6px;
    }
    .stExpander summary span { color: var(--muted) !important; font-size: 13px !important; }
    .stExpander summary:hover span { color: var(--text) !important; }
    .stExpander [data-testid="stExpanderToggleIcon"] { color: var(--muted) !important; }

    /* ═══ ALERTS ═════════════════════════════════════════ */
    [data-testid="stAlert"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--surface-3) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }
    [data-testid="stAlert"] p { color: var(--text) !important; }

    /* ═══ SPINNER ════════════════════════════════════════ */
    .stSpinner > div { color: var(--muted) !important; }

    /* ═══ CUSTOM COMPONENTS ═════════════════════════════ */
    .sv-welcome {
        display: flex; flex-direction: column;
        align-items: center; text-align: center;
        padding: 3rem 1rem 1rem;
    }
    .sv-welcome .sv-hero-eyebrow {
        display: inline-block; font-size: 11px; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
        color: var(--yo-yellow); margin-bottom: 12px; opacity: 0.9;
    }
    .sv-welcome h2 {
        font-size: 28px; font-weight: 700; line-height: 1.2;
        margin: 0 0 10px; max-width: 520px;
        color: var(--text) !important;
    }
    .sv-welcome .sv-duck-icon {
        font-size: 48px; margin-bottom: 8px;
    }
    .sv-welcome p {
        font-size: 15px; font-weight: 400; line-height: 1.55;
        color: var(--muted); max-width: 460px;
    }

    .sv-sidebar-brand { padding: 0 0 2px; margin-bottom: 2px; }
    .sv-sidebar-brand .sv-brand-kicker {
        font-size: 10px; font-weight: 700; letter-spacing: 0.25em;
        color: var(--yo-yellow); margin-bottom: 2px;
    }
    .sv-sidebar-brand .sv-brand-title {
        font-size: 17px; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--text) !important;
    }
    .sv-sidebar-brand .sv-brand-line {
        height: 2px; margin-top: 10px; border-radius: 2px;
        background: linear-gradient(90deg, var(--yo-yellow), var(--card-cyan) 50%, var(--card-lavender));
        opacity: 0.85;
    }

    .sv-page-hero { margin-bottom: 1.25rem; }
    .sv-page-hero h1, .sv-page-hero h2 { margin-bottom: 0.25rem !important; }
    .sv-page-hero p { color: var(--muted); font-size: 14px; line-height: 1.5; }

    .sv-live-dot {
        display: inline-block; width: 8px; height: 8px;
        background: var(--card-mint); border-radius: 50%;
        margin-right: 6px;
        animation: sv-pulse 2s ease-in-out infinite;
    }
    @keyframes sv-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.6; }
    }

    .sv-mode-label {
        font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px;
        color: rgba(255,255,255,0.35) !important;
        margin-bottom: 6px;
    }
    .sv-status-row {
        display: flex; align-items: center; gap: 6px;
        padding: 4px 0;
        font-size: 12px; font-weight: 400; color: var(--text);
    }

    .sv-flashcard {
        background: var(--surface-0);
        border: 1px solid var(--surface-2); border-radius: 16px;
        padding: 20px; margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .sv-flashcard-q { font-size: 15px; font-weight: 700; margin-bottom: 10px; }
    .sv-flashcard-a { font-size: 15px; font-weight: 400; color: var(--text); line-height: 22px; }
    .sv-flashcard-divider { border: none; border-top: 1px solid var(--surface-3); margin: 10px 0; }

    /* ═══ COLUMN ROWS ════════════════════════════════════ */
    [data-testid="stHorizontalBlock"] { min-width: 0 !important; }
    @media (max-width: 640px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0.75rem !important;
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"] {
            width: 100% !important; min-width: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            flex-direction: row !important; flex-wrap: wrap !important; gap: 0.5rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
            flex: 1 1 45% !important; min-width: 0 !important;
        }
    }

    /* ═══ RESPONSIVE ═════════════════════════════════════ */
    @media (max-width: 768px) {
        .block-container { max-width: 100% !important; padding: 0.75rem 0.75rem 5rem !important; }
        .sv-welcome h2 { font-size: 22px !important; }
        .sv-welcome { padding: 2rem 0.75rem 1rem; }
        [data-testid="stAppViewContainer"] h1 { font-size: 22px !important; }
        [data-testid="stAppViewContainer"] h2 { font-size: 18px !important; }
        [data-testid="stSidebar"] .stButton > button { min-height: 44px !important; }
    }
    @media (max-width: 480px) {
        .block-container { padding: 0.5rem 0.5rem 4.5rem !important; }
    }

    /* ═══ FOCUS VISIBLE ══════════════════════════════════ */
    button:focus-visible, input:focus-visible, textarea:focus-visible,
    [role="listbox"]:focus-visible, select:focus-visible {
        outline: 2px solid var(--yo-yellow) !important;
        outline-offset: 2px !important;
    }

    @media (prefers-reduced-motion: reduce) {
        .sv-live-dot { animation: none !important; }
        *, *::before, *::after { transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ── Connection check ─────────────────────────────────────────────────────────

client = get_client()
if client is None:
    st.error(
        "Cannot connect to VectorAI DB at localhost:50051. "
        "Make sure Docker is running and you ran `docker compose up`."
    )
    st.stop()

try:
    count = client.points.count(COLLECTION)
except Exception:
    count = 0

if count == 0:
    st.warning(
        "No course materials indexed yet. "
        "Drop your files (PDF, PPTX, DOCX) into the `materials/` folder and run:\n\n"
        "```\npython ingest.py\n```"
    )
    st.stop()


# ── Session state ────────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "duck_messages" not in st.session_state:
    st.session_state.duck_messages = []
if "exam_results" not in st.session_state:
    st.session_state.exam_results = None
if "flash_results" not in st.session_state:
    st.session_state.flash_results = None

use_llm = ollama_available()
courses = get_courses()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sv-sidebar-brand">
            <div class="sv-brand-kicker">Local · Private</div>
            <div class="sv-brand-title">StudyVector</div>
            <div class="sv-brand-line" aria-hidden="true"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Your notes. Your machine. No cloud.")
    st.markdown("---")

    # Mode nav
    st.markdown('<div class="sv-mode-label">MODE</div>', unsafe_allow_html=True)
    for key, label in MODES.items():
        icon = MODE_ICONS[key]
        if st.button(
            f"{icon}  {label}",
            key=f"mode_{key}",
            use_container_width=True,
            type="primary" if st.session_state.mode == key else "secondary",
        ):
            st.session_state.mode = key
            st.rerun()
    st.markdown("---")

    # Course filter
    st.markdown('<div class="sv-mode-label">COURSE FILTER</div>', unsafe_allow_html=True)
    course_options = ["All courses"] + courses
    selected_course = st.selectbox("Filter by course", course_options, label_visibility="collapsed")
    st.markdown("---")

    # Stats + Status combined
    col1, col2 = st.columns(2)
    col1.metric("Chunks", count)
    col2.metric("Courses", len(courses))

    if use_llm:
        st.markdown(
            f'<div class="sv-status-row"><span class="sv-live-dot"></span>{OLLAMA_MODEL}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sv-status-row" style="color:var(--muted)">LLM unavailable</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="sv-status-row"><span class="sv-live-dot"></span>VectorAI DB</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if st.session_state.confirm_clear:
        st.warning("Clear all conversations and results?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, clear", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.duck_messages = []
            st.session_state.exam_results = None
            st.session_state.flash_results = None
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()
    else:
        if st.button("CLEAR CONVERSATION", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: STUDY CHAT
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.mode == "chat":

    if not st.session_state.messages:
        st.markdown("""
        <div class="sv-welcome">
            <span class="sv-hero-eyebrow">RAG · Your materials</span>
            <h2>What do you want to study?</h2>
            <p>Ask a question and I'll find the answer across your course materials.</p>
        </div>
        """, unsafe_allow_html=True)
        pill_cols = st.columns(4)
        pill_labels = ["Summarise a topic", "Explain a concept", "Compare ideas", "Find definitions"]
        for col, label in zip(pill_cols, pill_labels):
            if col.button(label, key=f"pill_{label}", use_container_width=True, type="secondary"):
                st.session_state.messages.append({"role": "user", "content": label})
                st.rerun()

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "📚"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"**{s['source']}** p.{s['page']} ({s['course']}) · {s['score']:.3f}")
                        st.write(s["text"])
                        st.markdown("---")

    if query := st.chat_input("Ask anything about your course materials..."):
        st.chat_message("user", avatar="🧑‍🎓").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Searching your notes..."):
            results = search(query, selected_course)

        if not results:
            reply = "I couldn't find anything relevant. Try rephrasing or change the course filter."
            st.chat_message("assistant", avatar="📚").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            source_info = [
                {
                    "source": r.payload.get("source", "?"),
                    "page": r.payload.get("page", "?"),
                    "course": r.payload.get("course", "General"),
                    "score": r.score,
                    "text": r.payload.get("text", ""),
                }
                for r in results
            ]

            if use_llm:
                history_context = ""
                recent = st.session_state.messages[-6:]
                if len(recent) > 1:
                    history_context = "Previous conversation:\n"
                    for m in recent[:-1]:
                        role = "Student" if m["role"] == "user" else "Assistant"
                        history_context += f"{role}: {m['content']}\n"
                    history_context += "\n"

                prompt = build_rag_prompt(query, results)
                if history_context:
                    prompt = prompt.replace("Student question:", history_context + "Current question:")

                with st.chat_message("assistant", avatar="📚"):
                    streamed = st.write_stream(stream_ollama(prompt))
                    with st.expander(f"📄 {len(source_info)} sources", expanded=False):
                        for s in source_info:
                            st.caption(f"**{s['source']}** p.{s['page']} ({s['course']}) · {s['score']:.3f}")
                            st.write(s["text"])
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed,
                    "sources": source_info,
                })
            else:
                reply_parts = ["Here's what I found:\n"]
                for s in source_info[:3]:
                    reply_parts.append(f"**{s['source']}** p.{s['page']}:\n> {s['text'][:200]}...")
                reply = "\n\n".join(reply_parts)

                with st.chat_message("assistant", avatar="📚"):
                    st.markdown(reply)
                    with st.expander(f"📄 {len(source_info)} full excerpts", expanded=False):
                        for s in source_info:
                            st.caption(f"**{s['source']}** p.{s['page']} ({s['course']}) · {s['score']:.3f}")
                            st.write(s["text"])
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": source_info,
                })


# ══════════════════════════════════════════════════════════════════════════════
# MODE: EXAM PREP
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.mode == "exam":
    st.markdown(
        """
        <div class="sv-page-hero">
            <h2>Exam Prep</h2>
            <p>Paste an exam question — get a study guide built from your notes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exam_q = st.text_area(
        "Exam question",
        placeholder="e.g. 'Explain the process of meiosis and how it differs from mitosis.'",
        height=120,
        label_visibility="collapsed",
    )

    col_btn, col_k = st.columns([2, 1])
    with col_btn:
        go = st.button("Find study material", type="primary", use_container_width=True)
    with col_k:
        top_k = st.slider("Results", 3, 12, 8)

    if go and exam_q:
        with st.spinner("Scanning your notes..."):
            results = search(exam_q, selected_course, top_k=top_k)

        if not results:
            st.info("Nothing relevant found. Try different wording.")
        else:
            study_guide = None
            if use_llm:
                with st.spinner("Building study guide..."):
                    prompt = (
                        f"You are a study coach. A student needs to answer this exam question:\n\n"
                        f"\"{exam_q}\"\n\n"
                        f"Here are excerpts from their actual course notes:\n\n"
                        + "\n\n---\n\n".join(
                            f"[{r.payload.get('source','?')} p.{r.payload.get('page','?')}]\n{r.payload.get('text','')}"
                            for r in results
                        )
                        + "\n\nWrite a concise study guide (bullet points) that helps the student answer this exam question, drawing only from the excerpts above. Mention the source for each key point."
                    )
                    study_guide = ask_ollama(prompt)

            by_source: dict[str, list] = {}
            for r in results:
                src = r.payload.get("source", "Unknown") if r.payload else "Unknown"
                by_source.setdefault(src, []).append(r)

            # Serialize results for session state persistence
            source_data = {}
            for source, chunks in by_source.items():
                source_data[source] = [
                    {"page": r.payload.get("page", "?"), "score": r.score,
                     "course": r.payload.get("course", ""), "text": r.payload.get("text", "")}
                    for r in sorted(chunks, key=lambda x: x.payload.get("page", 0))
                ]

            st.session_state.exam_results = {
                "guide": study_guide,
                "sources": source_data,
            }

    # Display persisted results
    if st.session_state.exam_results:
        er = st.session_state.exam_results
        if er["guide"]:
            st.markdown("### Study Guide")
            st.markdown(er["guide"])
            st.markdown("---")
        st.markdown(f"### Source Material ({len(er['sources'])} file(s))")
        for source, chunks in er["sources"].items():
            with st.expander(f"**{source}** — {len(chunks)} excerpt(s)", expanded=True):
                for c in chunks:
                    st.markdown(f"**Page {c['page']}** · score {c['score']:.3f} · {c['course']}")
                    st.write(c["text"])
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: FLASHCARDS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.mode == "flashcards":
    st.markdown(
        """
        <div class="sv-page-hero">
            <h2>Flashcards</h2>
            <p>Generate Q&amp;A flashcards from your notes for active recall.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not use_llm:
        st.warning("Flashcards require Ollama to be running.")
    else:
        flash_topic = st.text_input(
            "Topic",
            placeholder='e.g. "Evolution of management theories"',
            key="flash_topic",
        )

        col_gen, col_num = st.columns([2, 1])
        with col_gen:
            gen = st.button("Generate Flashcards", type="primary", use_container_width=True)
        with col_num:
            num_cards = st.slider("Cards", 3, 10, 5, key="num_cards")

        if gen and flash_topic:
            with st.spinner("Finding relevant material..."):
                results = search(flash_topic, selected_course, top_k=8)

            if not results:
                st.info("No relevant material found for this topic.")
            else:
                context = "\n\n---\n\n".join(
                    f"[{r.payload.get('source','?')} p.{r.payload.get('page','?')}]\n{r.payload.get('text','')}"
                    for r in results
                )
                prompt = f"""You are a study assistant creating flashcards. Using ONLY the course material below, generate exactly {num_cards} flashcards as Q&A pairs.

Format each flashcard exactly like this:
Q: [question]
A: [concise answer]

Make questions that test understanding, not just recall. Cover different aspects of the topic.

Course material:
{context}

Topic: {flash_topic}

Flashcards:"""

                with st.spinner("Generating flashcards..."):
                    raw = ask_ollama(prompt)

                cards = []
                current_q = None
                for line in raw.split("\n"):
                    line = line.strip()
                    if line.startswith("Q:"):
                        current_q = line[2:].strip()
                    elif line.startswith("A:") and current_q:
                        cards.append({"q": current_q, "a": line[2:].strip()})
                        current_q = None

                if cards:
                    st.session_state.flash_results = cards
                else:
                    st.session_state.flash_results = raw

        # Display persisted flashcard results
        if st.session_state.flash_results:
            fr = st.session_state.flash_results
            if isinstance(fr, list):
                accent_colors = [
                    "var(--card-blue)", "var(--card-mint)",
                    "var(--card-cyan)", "var(--card-lavender)",
                ]
                for i, card in enumerate(fr):
                    accent = accent_colors[i % len(accent_colors)]
                    q_safe = html.escape(card['q'])
                    a_safe = html.escape(card['a'])
                    st.markdown(f"""
                    <div class="sv-flashcard">
                        <div class="sv-flashcard-q" style="color:{accent}">Q: {q_safe}</div>
                        <hr class="sv-flashcard-divider">
                        <div class="sv-flashcard-a">A: {a_safe}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(fr)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: RUBBER DUCK
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.mode == "duck":

    if not st.session_state.duck_messages:
        st.markdown("""
        <div class="sv-welcome">
            <span class="sv-hero-eyebrow">Socratic · No lectures</span>
            <div class="sv-duck-icon">🦆</div>
            <h2>Rubber Duck</h2>
            <p>Explain your problem out loud. The duck listens, asks questions, and helps you think it through. No judgement, just quacking good logic.</p>
        </div>
        """, unsafe_allow_html=True)
        duck_pill_cols = st.columns(4)
        duck_pill_labels = ["I'm stuck on...", "I don't understand...", "How does ... work?", "Help me think through..."]
        for col, label in zip(duck_pill_cols, duck_pill_labels):
            if col.button(label, key=f"duck_pill_{label}", use_container_width=True):
                st.session_state.duck_messages.append({"role": "user", "content": label})
                st.rerun()

    for msg in st.session_state.duck_messages:
        avatar = "🦆" if msg["role"] == "assistant" else "🧑‍🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if not use_llm:
        st.warning("Rubber Duck mode requires Ollama to be running.")
    elif duck_input := st.chat_input("Tell the duck what's confusing you..."):
        st.chat_message("user", avatar="🧑‍🎓").markdown(duck_input)
        st.session_state.duck_messages.append({"role": "user", "content": duck_input})

        results = search(duck_input, selected_course, top_k=4)
        note_context = ""
        if results:
            note_context = "\n\nRelevant excerpts from the student's notes (use these to ground your response):\n"
            note_context += "\n---\n".join(
                r.payload.get("text", "") for r in results if r.payload
            )

        duck_history = ""
        for m in st.session_state.duck_messages[-8:]:
            role = "Student" if m["role"] == "user" else "Duck"
            duck_history += f"{role}: {m['content']}\n"

        prompt = f"""You are a rubber duck study buddy. Your job is to help the student think through problems by:

1. Asking clarifying questions ("What do you mean by...?", "Can you break that down?")
2. Gently guiding them toward the answer without giving it away immediately
3. Encouraging them to explain concepts in their own words
4. Pointing out contradictions or gaps in their reasoning
5. Celebrating when they have breakthroughs

Be warm, casual, and encouraging. Use short responses. Ask ONE question at a time.
If the student is clearly stuck after trying, then give a helpful nudge with a hint from their notes.
Do NOT lecture. Do NOT dump information. Let THEM do the thinking.
{note_context}

Conversation so far:
{duck_history}

Duck:"""

        with st.chat_message("assistant", avatar="🦆"):
            streamed = st.write_stream(stream_ollama(prompt))

        st.session_state.duck_messages.append({
            "role": "assistant",
            "content": streamed,
        })
