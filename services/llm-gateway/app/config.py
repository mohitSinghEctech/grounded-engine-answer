from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Grounded Answer Engine"
    app_version: str = "0.2.0"

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Which provider, and which of its models. Both resolve through
    # app/vendors.py, which knows what each combination accepts; leave
    # the model unset to get that vendor's cheapest capable one.
    llm_vendor: str = "openai"
    llm_model: str | None = None
    llm_api_key: str

    # Optional override. Normally the vendor supplies its own endpoint.
    llm_base_url: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "low"

    request_timeout_seconds: float = 30.0

    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 5.0
    max_retry_time_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
