import logging
import time

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_client
from app.schemas import AskRequest, AskResponse
from app.services.base import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["ask"]
)

@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
    llm_client: LLMClient = Depends(get_llm_client),
):
    start = time.perf_counter()
    
    result = await llm_client.generate(
        prompt=request.question,
        max_tokens=request.max_tokens,
    )
    
    latency_ms = int(
        (time.perf_counter() - start)*1000
    )
    
    logger.info(
        "LLM request | question=%s | latency_ms=%s | "
        "prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
        request.question[:200],
        latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        result.total_tokens,
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