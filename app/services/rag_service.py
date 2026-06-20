import os
from typing import List, Tuple, Optional, Dict, Any

from app.services.vector_store import VectorStoreService
from app.core.config import settings


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _build_context(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        ts = _format_timestamp(doc["metadata"].get("start_time", 0))
        parts.append(f"[Passage {i} — at {ts}]\n{doc['text']}")
    return "\n\n".join(parts)


def _answer_with_gemini(question: str, context: str, title: str) -> str:
    """Use Gemini API to generate a proper answer from context."""
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return _fallback_answer(question, context)

        client = genai.Client(api_key=api_key)

        prompt = f"""You are an AI assistant that answers questions about YouTube videos based on their transcript.

Video Title: "{title}"

Transcript Passages:
{context}

User Question: {question}

Instructions:
- Answer the question clearly and helpfully based on the transcript passages above
- Be conversational and direct — like ChatGPT
- If the answer is in the transcript, explain it well in your own words
- Include specific details, examples, or steps mentioned in the video
- Keep the answer focused and concise (3-6 sentences usually)
- If the transcript doesn't fully answer the question, say what IS covered and what's missing
- Do NOT just copy-paste the transcript — summarize and explain it properly

Answer:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()

    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return _fallback_answer(question, context)


def _fallback_answer(question: str, context: str) -> str:
    """Fallback if Gemini fails — return best chunk."""
    lines = context.split("\n\n")
    best = lines[1] if len(lines) > 1 else context
    return f"Based on the video transcript:\n\n{best}"


class RAGService:
    def __init__(self):
        self.vs = VectorStoreService.get_instance()

    def answer(
        self,
        question: str,
        video_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        # Retrieve relevant chunks
        docs = self.vs.query(
            question=question,
            video_id=video_id,
            top_k=settings.TOP_K_RESULTS,
        )

        video_ids = list({d["metadata"].get("video_id", "") for d in docs})
        
        if not docs:
            return (
                "I couldn't find relevant information in the indexed videos. "
                "Try rephrasing your question or ingest a video on this topic first.",
                [],
                [],
            )

        # Build context from retrieved chunks
        title = docs[0]["metadata"].get("title", "the video")
        context = _build_context(docs)

        # Generate proper answer with Gemini
        answer = _answer_with_gemini(question, context, title)

        # Append timestamp jump link
        best_ts = int(docs[0]["metadata"].get("start_time", 0))
        best_vid = docs[0]["metadata"].get("video_id", "")
        answer += f"\n\n📺 Jump to: https://www.youtube.com/watch?v={best_vid}&t={best_ts}s"

        return answer, docs, video_ids
