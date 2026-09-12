import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import AppError
from app.logging_config import configure_logging
from app.middleware import RequestIDMiddleware, request_id_context
from app.routers import ask, health
from app.schemas import ErrorResponse
from app.services.gateway_client import HttpLLMGateway

logger = logging.getLogger(__name__)


async def _propagate_request_id(request: httpx.Request) -> None:
    request.headers["X-Request-ID"] = request_id_context.get()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info(
        "Application starting | llm_gateway_url=%s | corpus_date=%s",
        settings.llm_gateway_url,
        settings.corpus_date,
    )

    client = httpx.AsyncClient(
        base_url=settings.llm_gateway_url,
        timeout=settings.llm_gateway_timeout_seconds,
        event_hooks={"request": [_propagate_request_id]},
    )

    app.state.llm_gateway = HttpLLMGateway(client=client, settings=settings)

    yield

    logger.info("Application shutting down...")

    await client.aclose()


async def app_error_handler(request: Request, exc: AppError):
    request_id = getattr(request.state, "request_id", "-")

    headers = {}
    retry_after = getattr(exc, "retry_after_seconds", None)

    if retry_after is not None:
        headers["Retry-After"] = str(int(retry_after))

    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content=ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
        ).model_dump(),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
):
    request_id = getattr(request.state, "request_id", "-")

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="The request was invalid.",
            request_id=request_id,
            details=[
                {
                    "field": ".".join(str(p) for p in err["loc"]),
                    "message": err["msg"],
                }
                for err in exc.errors()
            ],
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "-")

    logger.exception("Unhandled exception")

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            request_id=request_id,
        ).model_dump(),
    )


def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging(settings)

    is_prod = settings.environment == "production"

    app = FastAPI(
        title=settings.app_name,
        description="Grounded answers about Indian income tax law",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
    )

    app.add_middleware(RequestIDMiddleware)

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(health.router)
    app.include_router(ask.router)

    return app