---
title: ClipWise
emoji: 🎬
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
---

# ClipWise — AI YouTube Learning Assistant

> RAG-powered Q&A over YouTube video transcripts. Ask anything — get answers with timestamps.

---

## Architecture

```
YouTube URL
    │
    ▼
youtube-transcript-api      ← fetches captions
    │
    ▼
Text Chunker (char-overlap)  ← splits into 800-char chunks with 150-char overlap
    │
    ▼
Embedding Backend            ← sentence-transformers/all-MiniLM-L6-v2 (or TF-IDF fallback)
    │
    ▼
ChromaDB (persistent)        ← stores vectors + metadata (video_id, title, start_time)
    │
    ▼
FastAPI REST API             ← /ingest, /query, /videos, /health
    │
    ▼
HTML/JS Frontend             ← served by FastAPI at /
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health + indexed video count |
| `POST` | `/ingest/` | Ingest a YouTube video (URL + optional title) |
| `POST` | `/query/` | Ask a question (optionally filter by video_id) |
| `GET` | `/videos/` | List all indexed videos |
| `DELETE` | `/videos/{video_id}` | Remove a video from the index |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `GET` | `/redoc` | ReDoc API documentation |

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python run.py
# or: uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 3. Open http://localhost:7860
```

## Deploy to HuggingFace Spaces

```bash
# Create a new Space (Docker SDK) and push:
git init
git add .
git commit -m "Initial ClipWise deployment"
git remote add space https://huggingface.co/spaces/<your-username>/clipwise
git push space main
```

## Project Structure

```
clipwise/
├── app/
│   ├── main.py               # FastAPI app factory + router registration
│   ├── core/
│   │   └── config.py         # Settings (Pydantic BaseSettings)
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── routers/
│   │   ├── ingest.py         # POST /ingest/
│   │   ├── query.py          # POST /query/
│   │   ├── videos.py         # GET/DELETE /videos/
│   │   └── health.py         # GET /health
│   └── services/
│       ├── transcript_service.py   # YouTube transcript fetching + chunking
│       ├── vector_store.py         # ChromaDB + embedding backend (pluggable)
│       └── rag_service.py          # RAG pipeline (retrieve + synthesize)
├── static/
│   └── index.html            # Frontend UI (served by FastAPI)
├── chroma_db/                # ChromaDB persistence (auto-created)
├── Dockerfile                # HuggingFace Spaces deployment
├── requirements.txt
├── run.py                    # Entrypoint
└── .env.example
```

## Embedding Backend

ClipWise auto-selects the best available embedding backend:

1. **`sentence-transformers/all-MiniLM-L6-v2`** — 384-dim dense embeddings (best semantic quality; requires internet on first run to download model)
2. **TF-IDF char-ngram fallback** — 512-dim sparse-ish embeddings (pure local, zero downloads, works offline)

On HuggingFace Spaces, backend #1 is used. The TF-IDF fallback kicks in automatically in offline/sandbox environments.

## Stack

- **FastAPI** — async REST API with auto OpenAPI docs
- **LangChain** — document abstractions, text splitters
- **ChromaDB** — persistent vector store with cosine similarity
- **sentence-transformers** — local HuggingFace embeddings
- **youtube-transcript-api** — transcript fetching (no YouTube API key needed)
- **scikit-learn** — TF-IDF fallback embeddings
- **Pydantic v2** — request/response validation
