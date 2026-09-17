import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import AppError
from app.logging_config import configure_logging
from app.middleware import RequestIDMiddleware, request_id_context
from app.retrieval import GroundedRetriever, embedders, stores
from app.routers import ask, health, search
from app.schemas import ErrorResponse
from app.services.gateway_client import HttpLLMGateway

logger = logging.getLogger(__name__)


async def _propagate_request_id(request: httpx.Request) -> None:
    request.headers["X-Request-ID"] = request_id_context.get()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info(
        "Application starting | llm_gateway_url=%s | embedder=%s | store=%s",
        settings.llm_gateway_url,
        settings.embedder_provider,
        settings.vector_store_provider,
    )

    client = httpx.AsyncClient(
        base_url=settings.llm_gateway_url,
        timeout=settings.llm_gateway_timeout_seconds,
        event_hooks={"request": [_propagate_request_id]},
    )

    # Both sides are chosen by name, so swapping either is a config
    # change. GroundedRetriever holds the policy and knows neither.
    retriever = GroundedRetriever(
        embedder=embedders.create(settings),
        store=stores.create(settings),
        settings=settings,
    )

    # Compose starts qdrant in parallel and ECS gives no ordering at all, so
    # the index may not answer on the first try. A RuntimeError is a config
    # error that retrying will never fix, so it is re-raised immediately.
    for attempt in range(5):
        try:
            await retriever.verify()
            break
        except RuntimeError:
            raise
        except Exception as exc:
            if attempt == 4:
                raise

            logger.warning(
                "Index not ready, retrying | attempt=%s | reason=%s",
                attempt + 1,
                type(exc).__name__,
            )
            await asyncio.sleep(2**attempt)

    app.state.llm_gateway = HttpLLMGateway(client=client, settings=settings)
    app.state.retriever = retriever

    yield

    logger.info("Application shutting down...")

    await client.aclose()
    # The retriever owns its store, so it owns closing it - this function no
    # longer knows whether there is a connection to close at all.
    await retriever.close()


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


async def validation_error_handler(request: Request, exc: RequestValidationError):
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

    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]

    if origins:
        # Credentials stay off. This service has no cookies and no session,
        # so a browser needs nothing but the response body - and allowing
        # credentials alongside "*" is rejected by browsers anyway.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

        logger.info("CORS enabled | origins=%s", origins)

    app.add_middleware(RequestIDMiddleware)

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(search.router)

    return app
