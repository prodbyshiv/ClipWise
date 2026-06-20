import os
from typing import List, Tuple, Optional, Dict, Any
from app.services.vector_store import VectorStoreService
from app.core.config import settings
from dotenv import load_dotenv
load_dotenv()

def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"

def _build_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        ts = _format_timestamp(doc["metadata"].get("start_time", 0))
        parts.append(f"[Passage {i} at {ts}]\n{doc['text']}")
    return "\n\n".join(parts)

def _answer_with_gemini(question, context, title):
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    print(f"[DEBUG] Key: {api_key[:10] if api_key else 'NOT FOUND'}")
    if not api_key:
        return "Gemini key nahi mili."
    client = genai.Client(api_key=api_key)
    prompt = (
        f'You are an AI assistant. Answer the question based on this YouTube video transcript.\n\n'
        f'Video: "{title}"\n\n'
        f'Transcript:\n{context}\n\n'
        f'Question: {question}\n\n'
        f'Give a clear, helpful answer in 4-6 sentences. Do not copy the transcript, explain it properly.'
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

class RAGService:
    def __init__(self):
        self.vs = VectorStoreService.get_instance()

    def answer(self, question, video_id=None):
        docs = self.vs.query(question=question, video_id=video_id, top_k=settings.TOP_K_RESULTS)
        video_ids = list({d["metadata"].get("video_id", "") for d in docs})
        if not docs:
            return ("No info found. Ingest a video first.", [], [])
        title = docs[0]["metadata"].get("title", "the video")
        context = _build_context(docs)
        answer = _answer_with_gemini(question, context, title)
        best_ts = int(docs[0]["metadata"].get("start_time", 0))
        best_vid = docs[0]["metadata"].get("video_id", "")
        answer += f"\n\n📺 https://www.youtube.com/watch?v={best_vid}&t={best_ts}s"
        return answer, docs, video_ids