from fastapi import APIRouter, HTTPException
from app.models.schemas import ListVideosResponse, VideoInfo, DeleteResponse
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/", response_model=ListVideosResponse, summary="List all indexed videos")
async def list_videos():
    """
    **List all YouTube videos currently indexed in ChromaDB.**

    Returns video metadata including title, chunk count, and original URL.
    """
    vs = VectorStoreService.get_instance()
    try:
        raw = vs.list_videos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

    videos = [VideoInfo(**v) for v in raw]
    return ListVideosResponse(videos=videos, total=len(videos))


@router.delete("/{video_id}", response_model=DeleteResponse, summary="Delete an indexed video")
async def delete_video(video_id: str):
    """
    **Remove a video and all its transcript chunks from the index.**

    This is permanent. Re-ingest the video to index it again.
    """
    vs = VectorStoreService.get_instance()

    if not vs.video_exists(video_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{video_id}' not found in index.",
        )

    success = vs.delete_video(video_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete video.")

    return DeleteResponse(
        video_id=video_id,
        message=f"Video '{video_id}' and all its chunks have been removed from the index.",
    )
