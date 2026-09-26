# Offline Multi-Document QA (RAG)

A local-first, offline-capable **Retrieval-Augmented Generation** app for asking questions about your own documents, such as long insurance policies. It combines lexical BM25-style ranking with semantic embeddings and stores documents and chunks in SQLite. It only returns answers that are grounded in retrieved passages, with citations.

📄 Project report: [`Implementation_of_RAG_Based_Question-Answering_Application.pdf`](data/uploads/1/Implementation_of_RAG_Based_Question-Answering_Application.pdf)

## Features

- Offline document ingestion for `.txt`, `.md`, `.pdf` and `.docx`
- User accounts with salted password hashing and per-user document collections
- Persistent SQLite storage for users, documents, chunks and sessions
- Hybrid retrieval that combines lexical and semantic scores
- Citation-based answers built from retrieved chunks
- Optional local LLM answers through [Ollama](https://ollama.com) (`phi3:mini` or `llama3.2:1b`), falling back to extractive answers when no model is available
- Minimal web UI for upload, search and answer review

## Getting started

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start the application:

   ```bash
   python run.py
   ```

3. Open http://127.0.0.1:8010, create an account, upload documents and ask questions.

If you prefer `uvicorn`, run it from this folder on a free port:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

### Optional: generated answers with Ollama

Install Ollama and pull one of the supported models:

```bash
ollama pull phi3:mini      # or: ollama pull llama3.2:1b
```

When Ollama is running on `127.0.0.1:11434`, answers are written by the local model and grounded in the retrieved citations. Without it, the app still works and returns extractive, cited answers.

## Running tests

```bash
pip install pytest httpx
python -m pytest tests
```

## Project structure

```
app/
├── main.py         # FastAPI routes (auth, upload, ask)
├── auth.py         # Salted password hashing & sessions
├── db.py           # SQLite schema and connections
├── documents.py    # Parsing and chunking of txt/md/pdf/docx
├── embeddings.py   # sentence-transformers embeddings (with offline fallback)
├── retrieval.py    # Hybrid BM25 + semantic ranking
├── qa.py           # Grounded answer assembly with citations
├── llm.py          # Optional Ollama generation
├── templates/      # Web UI
└── static/
data/               # SQLite DB and uploaded files (created at runtime)
tests/              # End-to-end tests with FastAPI TestClient
```

## Notes

- The app is designed for offline use once the dependencies and the embedding model are available locally.
- `sentence-transformers` tries to load `all-MiniLM-L6-v2`. For a fully offline setup, download that model on the target machine beforehand, or point the code to a locally cached copy.
- If the embedding model cannot be loaded, the app falls back to a deterministic local embedding approximation. Retrieval still works, with lower semantic quality.
- The database (`data/offline_qa.sqlite3`) is created automatically on first run. It is git-ignored so that accounts and session tokens are never committed.
