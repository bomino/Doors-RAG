"""Configuration module"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Database
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    redis_url: str = "localhost:6379"

    # Application
    environment: str = "development"
    log_level: str = "INFO"

    # RAG Settings
    max_chunk_size: int = 1000
    confidence_threshold: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()
