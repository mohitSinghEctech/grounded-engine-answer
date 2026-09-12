from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Tax Agent"
    app_version: str = "0.1.0"
    
    environment: Literal[
        "development",
        "test",
        "production"
    ] = "development"
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL"
    ] = "INFO"
    
    llm_gateway_url: str
    llm_gateway_timeout_seconds: float = 60.0
    
    corpus_date: str = "unknown"
    
    model_config = SettingsConfigDict(env_file=".env")
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
    