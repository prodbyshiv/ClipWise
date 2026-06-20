"""
ClipWise — Vector Store Service
================================
Uses ChromaDB for persistence + semantic retrieval.

Embedding backend (auto-selected):
  1. sentence-transformers/all-MiniLM-L6-v2  (best quality; default when HuggingFace is reachable)
  2. TF-IDF char-ngram (pure local fallback — no internet required; used in offline/sandbox environments)

On HuggingFace Spaces, backend #1 is used automatically.
In environments without internet access, #2 is used seamlessly.
"""

import os
import uuid
import pickle
import json
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.core.config import settings


# ── EMBEDDING BACKEND ─────────────────────────────────────────────────────────

class SentenceTransformerBackend:
    """sentence-transformers backend (online, best quality)."""

    def __init__(self, model_name: str):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.ef = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.dim = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.ef.embed_query(text)


class TfidfBackend:
    """
    Pure-local TF-IDF char-ngram embedding (no internet required).
    Uses character n-grams (2–4) for subword-level semantic similarity.
    Fitted incrementally as documents are added.
    """

    VOCAB_PATH = "./chroma_db/_tfidf_vocab.pkl"

    def __init__(self, n_features: int = 512):
        self.n_features = n_features
        self.dim = n_features
        self.vectorizer = TfidfVectorizer(
            max_features=n_features,
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
        )
        self._corpus: List[str] = []
        self._fitted = False
        self._load_vocab()

    def _load_vocab(self):
        """Load previously fitted vocabulary if available."""
        if os.path.exists(self.VOCAB_PATH):
            try:
                with open(self.VOCAB_PATH, "rb") as f:
                    state = pickle.load(f)
                self.vectorizer = state["vectorizer"]
                self._corpus = state.get("corpus", [])
                self._fitted = True
                print("[Embeddings] Loaded TF-IDF vocab from disk")
            except Exception as e:
                print(f"[Embeddings] Could not load vocab: {e}")

    def _save_vocab(self):
        os.makedirs(os.path.dirname(self.VOCAB_PATH), exist_ok=True)
        with open(self.VOCAB_PATH, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "corpus": self._corpus}, f)

    def fit_on(self, texts: List[str]):
        """Add new texts to the corpus and refit."""
        self._corpus.extend(texts)
        # Deduplicate
        self._corpus = list(dict.fromkeys(self._corpus))
        self.vectorizer.fit(self._corpus)
        self._fitted = True
        self._save_vocab()

    def _vectorize(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit_on(texts)
        X = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X / norms).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.fit_on(texts)  # always refit with new text
        return self._vectorize(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize([text])[0]


def _build_embedding_backend():
    """Try sentence-transformers, fall back to TF-IDF."""
    try:
        backend = SentenceTransformerBackend(settings.EMBEDDING_MODEL)
        print(f"[Embeddings] Using sentence-transformers: {settings.EMBEDDING_MODEL}")
        return backend
    except Exception as e:
        print(f"[Embeddings] sentence-transformers unavailable ({type(e).__name__}), using TF-IDF fallback")
        return TfidfBackend(n_features=512)


# ── CHROMA EMBEDDING FUNCTION ADAPTER ────────────────────────────────────────

class _ChromaEFAdapter:
    """Wraps our backend into ChromaDB's EmbeddingFunction interface."""

    def __init__(self, backend):
        self._backend = backend

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._backend.embed_documents(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        # ChromaDB calls embed_query with a list of query strings
        return self._backend.embed_documents(input)

    def name(self) -> str:
        return f"clipwise_{type(self._backend).__name__.lower()}"


# ── VECTOR STORE SERVICE ─────────────────────────────────────────────────────

class VectorStoreService:
    """
    Manages ChromaDB vector store for transcript chunks.
    Singleton — one instance per process.
    """

    _instance: Optional["VectorStoreService"] = None

    def __init__(self):
        self._backend = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _lazy_init(self):
        if self._initialized:
            return

        print("[VectorStore] Initializing…")
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        self._backend = _build_embedding_backend()

        self._chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        ef_adapter = _ChromaEFAdapter(self._backend)

        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=ef_adapter,
            metadata={"hnsw:space": "cosine"},
        )

        self._initialized = True
        print("[VectorStore] Ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def add_video(
        self,
        video_id: str,
        title: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Embed and index transcript chunks. Idempotent — re-ingest replaces."""
        self._lazy_init()
        self.delete_video(video_id)

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "video_id": video_id,
                "title": title,
                "start_time": float(c.get("start_time", 0.0)),
                "chunk_index": int(c.get("chunk_index", 0)),
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            }
            for c in chunks
        ]
        ids = [f"{video_id}_{c['chunk_index']}_{uuid.uuid4().hex[:6]}" for c in chunks]

        # Batch in groups of 100 to avoid memory spikes
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self._collection.add(
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

        print(f"[VectorStore] Indexed {len(texts)} chunks for '{title}' ({video_id})")
        return len(texts)

    def query(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Returns list of {text, metadata} dicts."""
        self._lazy_init()
        k = top_k or settings.TOP_K_RESULTS

        where = {"video_id": video_id} if video_id else None

        try:
            # For TF-IDF, refit includes the query text vocabulary
            if isinstance(self._backend, TfidfBackend):
                self._backend.fit_on([question])

            results = self._collection.query(
                query_texts=[question],
                n_results=min(k, self._collection.count() or 1),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            docs = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                docs.append({"text": doc, "metadata": meta, "score": 1 - dist})

            return docs

        except Exception as e:
            print(f"[VectorStore] Query error: {e}")
            return []

    def delete_video(self, video_id: str) -> bool:
        """Remove all chunks for a video."""
        self._lazy_init()
        try:
            existing = self._collection.get(where={"video_id": video_id})
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
                print(f"[VectorStore] Deleted {len(existing['ids'])} chunks for {video_id}")
                return True
        except Exception as e:
            print(f"[VectorStore] Delete error: {e}")
        return False

    def list_videos(self) -> List[Dict[str, Any]]:
        """Aggregate video metadata from all stored chunks."""
        self._lazy_init()
        try:
            all_docs = self._collection.get(include=["metadatas"])
            metadatas = all_docs.get("metadatas") or []

            video_map: Dict[str, Dict] = {}
            for meta in metadatas:
                vid = meta.get("video_id", "unknown")
                if vid not in video_map:
                    video_map[vid] = {
                        "video_id": vid,
                        "title": meta.get("title", "Untitled"),
                        "youtube_url": meta.get("youtube_url", f"https://www.youtube.com/watch?v={vid}"),
                        "chunk_count": 0,
                    }
                video_map[vid]["chunk_count"] += 1

            return list(video_map.values())
        except Exception as e:
            print(f"[VectorStore] List error: {e}")
            return []

    def video_exists(self, video_id: str) -> bool:
        self._lazy_init()
        try:
            result = self._collection.get(where={"video_id": video_id}, limit=1)
            return bool(result and result.get("ids"))
        except Exception:
            return False

    def get_indexed_count(self) -> int:
        return len(self.list_videos())
