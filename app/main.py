import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.routers import health, ask
from app.config import get_settings
from app.services.openai_compatible import OpenAICompatibleClient
from app.middleware import RequestIDMiddleware
from app.errors import AppError
from app.schemas import ErrorResponse
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Startup
    logger.info("Application starting...")

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )

    llm_client = OpenAICompatibleClient(client=client, settings=settings)

    app.state.llm_client = llm_client

    yield

    # Shutdown
    logger.info("Application shutting down...")

    await client.close()


async def app_error_handler(request: Request, exc: AppError):
    request_id = request.state.request_id

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.error_code, message=exc.message, request_id=request_id
        ).model_dump(),
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = request.state.request_id

    details = [
        {key: value for key, value in error.items() if key not in {"url", "ctx"}}
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="The request contains invalid data.",
            request_id=request_id,
            details=details,
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = request.state.request_id

    logger.exception(
        "Unhandled application exception", extra={"request_id": request_id}
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            request_id=request_id,
        ).model_dump(),
    )


def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description="AI Application for generating grounded answers",
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(health.router)
    app.include_router(ask.router)

    return app
