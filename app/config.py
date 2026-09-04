from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Grounded Answer Engine"
    app_version:str = "0.1.0"
    environment:str = "development"
    log_level: str = "INFO"
    
    # gemini_api_key: str
    # openai_api_key: str
    # groke_api_key: str
    
    model_config = SettingsConfigDict(
        env_file=".env"
    )
    
@lru_cache
def get_settings() -> Settings:
    return Settings()