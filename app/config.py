from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Grounded Answer Engine"
    app_version:str = "0.1.0"
    environment:str = "development"
    log_level: str = "INFO"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.7-flash"
    llm_api_key: str
    llm_base_url: str
    
    model_config = SettingsConfigDict(
        env_file=".env"
    )
    
@lru_cache
def get_settings() -> Settings:
    return Settings()