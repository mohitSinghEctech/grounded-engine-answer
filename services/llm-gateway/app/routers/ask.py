import logging
import time

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_client
from app.schemas import AskRequest, AskResponse, ErrorResponse, GenerateRequest
from app.services.base import LLMClient

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
    llm_client: LLMClient = Depends(get_llm_client),
):
    logger.info("Ask request started")
    start = time.perf_counter()

    result = await llm_client.generate(
        prompt=request.question,
        max_tokens=request.max_tokens,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "LLM request completed | "
        "model=%s | "
        "latency_ms=%s | "
        "prompt_tokens=%s | "
        "completion_tokens=%s | "
        "reasoning_tokens=%s | "
        "total_tokens=%s | "
        "finish_reason=%s",
        result.model,
        latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        result.reasoning_tokens,
        result.total_tokens,
        result.finish_reason,
    )

    return AskResponse(
        answer=result.text,
        model=result.model,
        latency_ms=latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
    )


@router.post(
    "/generate",
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
async def generate(
    request: GenerateRequest,
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Send an assembled prompt to the model.

    Same behaviour as /ask, but typed for a prompt rather than a user's
    question: no 2000-character limit, because callers of this endpoint
    build prompts from retrieved documents.
    """
    return await ask(
        request=AskRequest.model_construct(
            question=request.prompt,
            max_tokens=request.max_tokens,
        ),
        llm_client=llm_client,
    )
