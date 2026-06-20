from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "ClipWise"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI YouTube Learning Assistant — RAG-powered Q&A over video transcripts"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "clipwise_transcripts"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5

    # Gemini API
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
