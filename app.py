"""
StudyVector — Offline Study Assistant
Ask questions across all your course materials. Works offline. Nothing leaves your machine.

Run:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from actian_vectorai import VectorAIClient, Field, FilterBuilder
except ImportError:
    st.error("actian-vectorai not installed. Run: pip install actian_vectorai-0.1.0b2-py3-none-any.whl")
    st.stop()

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION  = "studyvector"
EMBED_MODEL = "all-MiniLM-L6-v2"
SERVER      = "localhost:50051"
TOP_K       = 6
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Connecting to VectorAI DB...")
def get_client():
    try:
        c = VectorAIClient(SERVER)
        c.connect()
        c.health_check()
        return c
    except Exception as e:
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


def get_courses() -> list[str]:
    client = get_client()
    if client is None:
        return []
    try:
        # Scroll a sample to collect course values (no field index in beta)
        results = client.points.search(
            COLLECTION,
            vector=[0.0] * 384,
            limit=200,
        )
        courses = sorted({r.payload.get("course", "General") for r in results if r.payload})
        return courses
    except Exception:
        return []


# ── Ollama LLM (optional, fully offline) ────────────────────────────────────

def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def ask_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(Ollama error: {e})"


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


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StudyVector",
    page_icon="📚",
    layout="wide",
)

# Header
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("📚 StudyVector")
    st.caption("Your course materials. Your laptop. No cloud.")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Offline ✓")

# Connection check
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
        "Drop your PDFs into the `materials/` folder and run:\n\n"
        "```\npython ingest.py\n```"
    )
    st.stop()

# Sidebar: course filter
st.sidebar.header("Filter by course")
courses = get_courses()
course_options = ["All courses"] + courses
selected_course = st.sidebar.selectbox("Course", course_options)

st.sidebar.markdown("---")
st.sidebar.metric("Chunks indexed", count)
st.sidebar.markdown(f"**Courses found:** {len(courses)}")
for c in courses:
    st.sidebar.markdown(f"- {c}")

use_llm = ollama_available()
st.sidebar.markdown("---")
if use_llm:
    st.sidebar.success(f"LLM: {OLLAMA_MODEL} (Ollama)")
else:
    st.sidebar.info("LLM: Not available\n\nInstall Ollama + llama3.2 for AI summaries.\nRetrieval still works without it.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_search, tab_exam = st.tabs(["Search your notes", "Exam prep mode"])


# ── Tab 1: Search ─────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Ask anything about your course material")
    query = st.text_input(
        "Question",
        placeholder='e.g. "What is the difference between TCP and UDP?"',
        label_visibility="collapsed",
    )

    if query:
        with st.spinner("Searching..."):
            results = search(query, selected_course)

        if not results:
            st.info("No relevant material found. Try rephrasing or select a different course filter.")
        else:
            if use_llm:
                with st.spinner("Generating answer with local LLM..."):
                    prompt = build_rag_prompt(query, results)
                    answer = ask_ollama(prompt)
                st.markdown("### Answer")
                st.markdown(answer)
                st.markdown("---")

            st.markdown(f"### {len(results)} most relevant excerpts")
            for i, r in enumerate(results, 1):
                p = r.payload or {}
                with st.expander(
                    f"**{p.get('source', 'Unknown')}** — page {p.get('page', '?')} "
                    f"({p.get('course', 'General')})  |  score: {r.score:.3f}",
                    expanded=(i == 1),
                ):
                    st.write(p.get("text", ""))


# ── Tab 2: Exam prep ─────────────────────────────────────────────────────────
with tab_exam:
    st.subheader("Paste an exam question — find everything you need to study")
    st.caption("Drop in a past paper question and StudyVector retrieves all the relevant material from your notes.")

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
            if use_llm:
                with st.spinner("Building personalised study summary..."):
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
                    summary = ask_ollama(prompt)
                st.markdown("### Study guide for this question")
                st.markdown(summary)
                st.markdown("---")

            # Group results by source file
            by_source: dict[str, list] = {}
            for r in results:
                src = r.payload.get("source", "Unknown") if r.payload else "Unknown"
                by_source.setdefault(src, []).append(r)

            st.markdown(f"### Source material ({len(by_source)} file(s))")
            for source, chunks in by_source.items():
                with st.expander(f"**{source}** — {len(chunks)} relevant excerpt(s)", expanded=True):
                    for r in sorted(chunks, key=lambda x: x.payload.get("page", 0)):
                        p = r.payload or {}
                        st.markdown(
                            f"**Page {p.get('page','?')}** · score {r.score:.3f} · {p.get('course','')}"
                        )
                        st.write(p.get("text", ""))
                        st.markdown("---")

            st.caption(
                f"Covers {len(set(r.payload.get('course','') for r in results if r.payload))} course(s): "
                + ", ".join(sorted({r.payload.get("course","") for r in results if r.payload}))
            )
