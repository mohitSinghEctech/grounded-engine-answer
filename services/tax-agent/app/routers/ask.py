import logging
import time

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_llm_gateway
from app.schemas import AskRequest, AskResponse, ErrorResponse
from app.services.base import LLMGateway

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask(
    request: AskRequest,
    gateway: LLMGateway = Depends(get_llm_gateway),
    settings: Settings = Depends(get_settings),
):
    logger.info("Ask request started")
    start = time.perf_counter()

    # Step 4 inserts retrieval here; step 5 replaces this with a
    # grounded prompt built from the retrieved passages.
    result = await gateway.generate(
        prompt=request.question,
        max_tokens=request.max_tokens,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "Ask request completed | "
        "model=%s | "
        "latency_ms=%s | "
        "prompt_tokens=%s | "
        "completion_tokens=%s | "
        "total_tokens=%s | "
        "finish_reason=%s",
        result.model,
        latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        result.total_tokens,
        result.finish_reason,
    )

    return AskResponse(
        answer=result.text,
        corpus_date=settings.corpus_date,
        model=result.model,
        latency_ms=latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
    )