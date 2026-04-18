# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

StudyVector is an offline RAG study assistant. Users drop PDFs into `materials/`, ingest them into Actian VectorAI DB, and query them via a Streamlit UI. An optional local LLM (Ollama) synthesises answers from retrieved chunks.

## Commands

```bash
# Start the vector database (required first)
docker compose up -d

# Activate venv
source .venv/bin/activate

# Install dependencies
pip install actian_vectorai-0.1.0b2-py3-none-any.whl
pip install -r requirements.txt

# Ingest PDFs into the database
python ingest.py                    # incremental — skips already-indexed files
python ingest.py --reset            # wipe and re-ingest from scratch
python ingest.py --dir ~/some/path  # custom folder

# Run the app
streamlit run app.py
```

## Architecture

Two Python files, no package structure:

- **`ingest.py`** — CLI pipeline: finds PDFs in `materials/`, extracts text with pdfplumber, splits into overlapping character-based chunks (400 chars, 80 overlap), embeds with `all-MiniLM-L6-v2` (384-dim), upserts into VectorAI DB. Tracks ingested files by MD5 hash in `.ingest_progress.json` to enable incremental runs. Creates fresh gRPC connections per file to avoid keepalive timeouts.

- **`app.py`** — Streamlit UI with two tabs: search and exam prep. Embeds the query, does cosine similarity search against VectorAI DB, shows top-k chunks. If Ollama is running, builds a RAG prompt and streams an LLM answer. Course filtering uses folder structure from ingestion (parent folder = course name).

Key shared constants between the two files: `COLLECTION = "studyvector"`, `EMBED_MODEL = "all-MiniLM-L6-v2"`, `SERVER = "localhost:50051"`. These must stay in sync.

## External dependencies

- **Actian VectorAI DB**: runs in Docker on port 50051 (gRPC). Client installed from bundled `.whl` file.
- **Ollama** (optional): HTTP API at `localhost:11434`. Model configured as `qwen3:8b` in app.py (README says `llama3.2` — the code takes precedence).
- **sentence-transformers**: `all-MiniLM-L6-v2` is downloaded on first run, then cached locally.
