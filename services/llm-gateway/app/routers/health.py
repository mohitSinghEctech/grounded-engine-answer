from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health(
    settings: Settings = Depends(get_settings),
):
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )
