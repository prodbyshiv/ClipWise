from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """**System health check.** Returns app status and indexed video count."""
    vs = VectorStoreService.get_instance()
    try:
        count = vs.get_indexed_count()
    except Exception:
        count = 0

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        embedding_model=settings.EMBEDDING_MODEL,
        indexed_videos=count,
    )
