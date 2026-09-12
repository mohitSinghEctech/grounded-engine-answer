from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Tax Agent"
    app_version: str = "0.1.0"

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_gateway_url: str
    llm_gateway_timeout_seconds: float = 60.0

    corpus_date: str = "unknown"

    qdrant_url: str
    qdrant_collection: str = "ita_sections"

    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    embedding_api_key: str
    embedding_base_url: str | None = None

    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.30

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
