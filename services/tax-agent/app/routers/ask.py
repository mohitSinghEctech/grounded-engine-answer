import logging
import time

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_llm_gateway, get_retriever
from app.prompt import REFUSAL, build_prompt, split_citations
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ErrorResponse,
    RefusalReason,
)
from app.scope import split_marker
from app.services.base import LLMGateway, Passage, Retriever
from app.tax_year import extract_tax_year

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _resolve_tax_year(request: AskRequest) -> tuple[int | None, str]:
    if request.tax_year is not None:
        return request.tax_year, "request"

    detected = extract_tax_year(request.question)

    return detected, ("question" if detected else "none")


def _corpus_date(passages: list[Passage], settings: Settings) -> str:
    # The date belongs to the data that was retrieved, not to this process,
    # so it stays true even if the service outlives the index that built it.
    for passage in passages:
        if passage.corpus_date and passage.corpus_date != "unknown":
            return passage.corpus_date

    return settings.corpus_date


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
    retriever: Retriever = Depends(get_retriever),
    gateway: LLMGateway = Depends(get_llm_gateway),
    settings: Settings = Depends(get_settings),
):
    started = time.perf_counter()

    tax_year, tax_year_source = _resolve_tax_year(request)

    logger.info(
        "Ask started | tax_year=%s | source=%s | top_k=%s",
        tax_year,
        tax_year_source,
        settings.retrieval_top_k,
    )

    passages = await retriever.search(
        question=request.question,
        top_k=settings.retrieval_top_k,
        tax_year=tax_year,
    )

    retrieval_ms = int((time.perf_counter() - started) * 1000)

    # Refusal is structural: with nothing retrieved the model is never called,
    # so it cannot be talked into answering from memory.
    if not passages:
        logger.info(
            "Ask refused, nothing retrieved | retrieval_ms=%s | tax_year=%s",
            retrieval_ms,
            tax_year,
        )

        return AskResponse(
            answer=REFUSAL,
            refused=True,
            refusal_reason="nothing_retrieved",
            citations=[],
            corpus_date=settings.corpus_date,
            tax_year=tax_year,
            tax_year_source=tax_year_source,
            retrieved=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retrieval_ms=retrieval_ms,
        )

    generation_started = time.perf_counter()

    result = await gateway.generate(
        prompt=build_prompt(request.question, passages),
        max_tokens=request.max_tokens,
    )

    llm_ms = int((time.perf_counter() - generation_started) * 1000)

    # Rule 3's signal, taken off the front of the answer before anything
    # else reads it. The model marks the question; this service decides what
    # that means.
    out_of_scope, answer_text = split_marker(result.text)

    cited, unsupported = split_citations(answer_text, passages)

    if unsupported:
        # The model cited a provision it was never given. Harmless to the
        # caller, but the first sign the prompt is losing control of it.
        logger.warning(
            "Unsupported citations in answer | count=%s | refs=%s",
            len(unsupported),
            sorted(f"{act} s.{number}" for act, number in unsupported),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)

    if result.finish_reason == "length":
        # Reasoning tokens count against max_tokens, so a model that thinks
        # hard can exhaust the budget before it finishes writing.
        logger.warning(
            "Answer truncated | completion_tokens=%s | reasoning_tokens=%s | "
            "raise max_tokens",
            result.completion_tokens,
            result.reasoning_tokens,
        )

    # Ordered by what the caller most needs to know. Out of scope outranks
    # ungrounded: a question that should not be answered was not answered
    # badly, and the two call for different fixes - a prompt change versus a
    # retrieval change.
    refusal_reason: RefusalReason = (
        "out_of_scope" if out_of_scope else "none" if cited else "not_grounded"
    )
    refused = refusal_reason != "none"

    logger.info(
        "Ask completed | refused=%s | refusal_reason=%s | retrieved=%s | cited=%s | "
        "unsupported=%s | retrieval_ms=%s | llm_ms=%s | latency_ms=%s | "
        "model=%s | total_tokens=%s | finish_reason=%s",
        refused,
        refusal_reason,
        len(passages),
        len(cited),
        len(unsupported),
        retrieval_ms,
        llm_ms,
        latency_ms,
        result.model,
        result.total_tokens,
        result.finish_reason,
    )

    return AskResponse(
        answer=answer_text,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=[]
        if out_of_scope
        else [
            Citation(
                act=p.act,
                section_number=p.section_number,
                section_title=p.section_title,
                page_start=p.page_start,
                score=p.score,
            )
            for p in cited
        ],
        corpus_date=_corpus_date(passages, settings),
        tax_year=tax_year,
        tax_year_source=tax_year_source,
        retrieved=len(passages),
        latency_ms=latency_ms,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
    )
