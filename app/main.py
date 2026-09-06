import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from app.routers import health, ask
from app.config import get_settings
from app.services.openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
        
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    for name in {
        "fastapi",
        "httpx",
        "httpx2",
        "openai",
        "httpcore",
        "httpcore2"
    }:
        logging.getLogger(name).setLevel(settings.log_level)
    
        
    # Startup
    logger.info("Application starting...")
    
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )
    
    llm_client = OpenAICompatibleClient(
        client=client,
        settings=settings
    )
    
    app.state.llm_client = llm_client
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    
    await client.close()

def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="AI Application for generating grounded answers",
        version=settings.app_version,
        lifespan=lifespan
    )
    
    app.include_router(health.router)
    app.include_router(ask.router)
    return app