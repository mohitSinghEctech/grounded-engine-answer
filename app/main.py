import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import health
from app.config import get_settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting...")
    
    app.state.gemini_client = "GEMINI CLIENT"
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    
    app.state.gemini_client = None

def create_app() -> FastAPI:
    settings = get_settings()
    
    logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    app = FastAPI(
        title=settings.app_name,
        description="AI Application for generating grounded answers",
        version=settings.app_version,
        lifespan=lifespan
    )
    
    app.include_router(health.router)
    return app