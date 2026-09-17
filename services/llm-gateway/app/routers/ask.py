import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_client
from app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    GenerateRequest,
    ToolCallOut,
)
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
    return await _complete(
        llm_client=llm_client,
        max_tokens=request.max_tokens,
        prompt=request.question,
    )


async def _complete(
    llm_client: LLMClient,
    max_tokens: int,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AskResponse:
    """One model call, logged and shaped. Both endpoints go through here."""
    logger.info(
        "Ask request started | turns=%s | tools=%s",
        len(messages) if messages else 1,
        len(tools) if tools else 0,
    )
    start = time.perf_counter()

    result = await llm_client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools or (),
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
        "finish_reason=%s | "
        "tool_calls=%s",
        result.model,
        latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        result.reasoning_tokens,
        result.total_tokens,
        result.finish_reason,
        [call.name for call in result.tool_calls],
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
        tool_calls=[
            ToolCallOut(id=call.id, name=call.name, arguments=call.arguments)
            for call in result.tool_calls
        ],
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
    """Send an assembled prompt, or a whole conversation with tools.

    Same behaviour as /ask, but typed for a prompt rather than a user's
    question: no 2000-character limit, because callers of this endpoint
    build prompts from retrieved documents. Pass `messages` + `tools`
    instead of `prompt` to run a step of a tool-calling loop.
    """
    return await _complete(
        llm_client=llm_client,
        max_tokens=request.max_tokens,
        prompt=request.prompt,
        messages=request.messages,
        tools=request.tools,
    )
