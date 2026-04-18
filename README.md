# StudyVector

> Ask questions across 4 semesters of course materials. Works offline. Nothing leaves your laptop.

Built on [Actian VectorAI DB](https://github.com/hackmamba-io/actian-vectorAI-db-beta).

---

## Setup (5 minutes)

### 1. Start VectorAI DB
```bash
docker compose up -d
```

### 2. Install Python dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install actian_vectorai-0.1.0b2-py3-none-any.whl   # from the beta repo
pip install -r requirements.txt
```

### 3. Add your course materials
Drop PDFs into `materials/`. Organise by course for best filtering:
```
materials/
  CS101/  lecture01.pdf  lecture02.pdf  ...
  MATH201/ calc_notes.pdf  past_paper_2023.pdf
  BIO301/  ...
```

### 4. Ingest
```bash
python ingest.py
# Re-run anytime you add more PDFs — it skips already-indexed files.
# To wipe and start fresh: python ingest.py --reset
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Optional: Local LLM (fully offline AI answers)

Install [Ollama](https://ollama.com) then:
```bash
ollama pull llama3.2
```
The app detects Ollama automatically. If it's not running, retrieval still works — you just get raw excerpts instead of synthesised answers.

---

## Offline verification

Turn off wifi. Run `streamlit run app.py`. Ask a question. It works.

VectorAI DB runs in Docker on your machine. The embedding model (`all-MiniLM-L6-v2`) is cached locally by sentence-transformers after the first download. Ollama is local. Zero network calls during use.

---

## Architecture

```
PDFs → pdfplumber (text extract) → chunker → sentence-transformers (embed)
     → Actian VectorAI DB (store + index)
     
Query → sentence-transformers (embed) → VectorAI DB (cosine search)
      → top-k chunks → Streamlit UI
      → (optional) Ollama LLM → synthesised answer
```

*Built for the Actian VectorAI DB Hackathon.*
