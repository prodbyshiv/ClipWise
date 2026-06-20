from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/query", tags=["Query"])

rag = RAGService()


@router.post("/", response_model=QueryResponse, summary="Ask a question about ingested videos")
async def query_videos(request: QueryRequest):
    """
    **Ask a question about ingested YouTube videos.**

    - Performs semantic search over stored transcript chunks
    - Returns a synthesized answer with source passages and timestamps
    - Optionally filter by `video_id` to restrict to one video
    - Leave `video_id` null to search across all indexed videos
    """
    vs = VectorStoreService.get_instance()

    if request.video_id and not vs.video_exists(request.video_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{request.video_id}' is not indexed. Ingest it first via /ingest.",
        )

    if vs.get_indexed_count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No videos indexed yet. Use /ingest to add a YouTube video first.",
        )

    try:
        answer, docs, video_ids = rag.answer(
            question=request.question,
            video_id=request.video_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    sources = [
        SourceChunk(
            video_id=doc["metadata"].get("video_id", ""),
            title=doc["metadata"].get("title", ""),
            text=doc["text"],
            start_time=doc["metadata"].get("start_time"),
            chunk_index=doc["metadata"].get("chunk_index", 0),
        )
        for doc in docs
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        video_ids_searched=video_ids,
    )
