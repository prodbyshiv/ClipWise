from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import IngestRequest, IngestResponse
from app.services.transcript_service import (
    extract_video_id,
    fetch_transcript,
    transcript_to_chunks_with_timing,
)
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("/", response_model=IngestResponse, summary="Ingest a YouTube video")
async def ingest_video(request: IngestRequest):
    """
    **Ingest a YouTube video for Q&A.**

    - Extracts the transcript from the YouTube URL
    - Chunks the transcript with overlap
    - Embeds chunks using sentence-transformers
    - Stores in ChromaDB for semantic retrieval

    Re-ingesting the same video replaces the previous index.
    """
    # Extract video ID
    video_id = extract_video_id(request.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Could not extract video ID from the provided URL.")

    # Fetch transcript
    try:
        transcript = fetch_transcript(video_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript is empty for this video.")

    # Chunk with timing metadata
    chunks = transcript_to_chunks_with_timing(
        transcript,
        chunk_size_chars=settings.CHUNK_SIZE * 2,  # chars, not tokens
        overlap_chars=settings.CHUNK_OVERLAP * 2,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="Failed to create chunks from transcript.")

    # Determine title
    title = request.video_title or f"YouTube Video ({video_id})"

    # Store in vector DB
    vs = VectorStoreService.get_instance()
    try:
        count = vs.add_video(video_id=video_id, title=title, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index video: {str(e)}")

    return IngestResponse(
        video_id=video_id,
        title=title,
        chunk_count=count,
        message=f"Successfully indexed '{title}' with {count} chunks. You can now ask questions about it.",
    )
