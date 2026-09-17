import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.dependencies import get_llm_gateway, get_retriever
from app.errors import AppError
from app.graph.build import run_graph_pipeline
from app.pipeline import steps
from app.progress import Progress, Step
from app.schemas import AskRequest, AskResponse, ErrorResponse
from app.services.base import LLMGateway, Retriever

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


async def run_linear_pipeline(
    request: AskRequest,
    retriever: Retriever,
    gateway: LLMGateway,
    settings: Settings,
    progress: Progress | None = None,
) -> AskResponse:
    """Retrieve, ground, generate, verify - in that order, no going back.

    Every stage is a call into `app/pipeline/steps.py`, which the graph in
    `app/graph/` also calls. This function contributes only the order and
    the one early exit, so it stays readable as the shape of the pipeline
    rather than its implementation.
    """
    context = steps.Context.open(
        request=request,
        retriever=retriever,
        gateway=gateway,
        settings=settings,
        progress=progress,
    )

    await steps.announce(context)

    year = steps.resolve_tax_year(request)

    await steps.report_year(context, year)

    retrieval = await steps.retrieve(context, year)

    # Refusal is structural: with nothing retrieved the model is never
    # called, so it cannot be talked into answering from memory.
    if not retrieval.passages:
        return await steps.refuse_nothing_retrieved(context, year, retrieval)

    prompt = await steps.make_prompt(context, retrieval.passages)

    result, llm_ms = await steps.generate(context, prompt)

    verification = await steps.verify(context, result, retrieval.passages)

    return await steps.finalise(context, year, retrieval, result, verification, llm_ms)


async def run_pipeline(
    request: AskRequest,
    retriever: Retriever,
    gateway: LLMGateway,
    settings: Settings,
    progress: Progress | None = None,
) -> AskResponse:
    """One question, one answer, whichever orchestrator is configured.

    The only implementation of /ask. The streaming endpoint passes a
    listening `Progress` and the plain one passes nothing, so neither can
    drift from the other as the pipeline changes.
    """
    if settings.pipeline == "graph":
        return await run_graph_pipeline(
            request, retriever, gateway, settings, progress=progress
        )

    return await run_linear_pipeline(
        request, retriever, gateway, settings, progress=progress
    )


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
    """One question, one JSON answer. Nothing is reported while it works."""
    return await run_pipeline(request, retriever, gateway, settings)


async def _stream(
    request: AskRequest,
    retriever: Retriever,
    gateway: LLMGateway,
    settings: Settings,
) -> AsyncIterator[str]:
    """Server-Sent Events: every step as it happens, then the answer.

    The pipeline runs as a separate task and pushes steps onto a queue;
    this generator drains the queue and writes frames. That is what lets
    the steps leave the building while the work is still going on - an
    ordinary `await` would finish everything before yielding anything.

    The answer itself is NOT streamed. Citations can only be checked once
    the text is complete, and around two answers in forty try to cite a
    provision that does not exist, so streaming the prose would mean
    showing a reader a fabricated section and retracting it afterwards.
    """
    queue: asyncio.Queue[Step | None] = asyncio.Queue()
    progress = Progress(queue=queue)

    async def run() -> AskResponse:
        try:
            return await run_pipeline(
                request, retriever, gateway, settings, progress=progress
            )
        finally:
            # Always close the queue, so a failure cannot hang the stream.
            await progress.finish()

    task = asyncio.create_task(run())

    while True:
        step = await queue.get()

        if step is None:
            break

        yield step.to_sse()

    try:
        response = await task
    except AppError as exc:
        # The same error vocabulary the JSON endpoint uses. A stream has
        # already sent HTTP 200, so the failure has to arrive as an event.
        yield Step(
            "failed",
            {"error_code": exc.error_code, "message": exc.message},
        ).to_sse(event="error")
        return
    except Exception:
        logger.exception("Streaming ask failed")
        yield Step(
            "failed",
            {"error_code": "INTERNAL_ERROR", "message": "Something went wrong."},
        ).to_sse(event="error")
        return

    yield Step("done", response.model_dump()).to_sse(event="result")


@router.post(
    "/ask/stream",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Server-Sent Events. One `status` frame per pipeline step, "
                "then a single `result` frame carrying the same body /ask "
                "returns, or an `error` frame."
            ),
        },
        422: {"model": ErrorResponse},
    },
)
async def ask_stream(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    gateway: LLMGateway = Depends(get_llm_gateway),
    settings: Settings = Depends(get_settings),
):
    return StreamingResponse(
        _stream(request, retriever, gateway, settings),
        media_type="text/event-stream",
        headers={
            # Without these a proxy may buffer the whole response and
            # deliver it at the end, which defeats the point entirely.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
