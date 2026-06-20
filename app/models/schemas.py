from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional
import re


class IngestRequest(BaseModel):
    youtube_url: str
    video_title: Optional[str] = None

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]+)",
            r"(?:https?://)?youtu\.be/([\w-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/([\w-]+)",
        ]
        for pattern in patterns:
            if re.search(pattern, v):
                return v
        raise ValueError("Must be a valid YouTube URL")


class IngestResponse(BaseModel):
    video_id: str
    title: str
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    question: str
    video_id: Optional[str] = None  # None = search across all ingested videos

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SourceChunk(BaseModel):
    video_id: str
    title: str
    text: str
    start_time: Optional[float] = None
    chunk_index: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    video_ids_searched: List[str]


class VideoInfo(BaseModel):
    video_id: str
    title: str
    chunk_count: int
    youtube_url: str


class ListVideosResponse(BaseModel):
    videos: List[VideoInfo]
    total: int


class DeleteResponse(BaseModel):
    video_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    indexed_videos: int
