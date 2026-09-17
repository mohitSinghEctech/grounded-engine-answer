"""One function per stage. No stage decides what runs next.

Every stage takes what it needs, reports what it did to `Progress`, and
returns data. The decisions - refuse or carry on, retry or accept - live
with the orchestrator, either the straight line in `routers/ask.py` or the
graph in `app/graph/`.

Extracted from `run_pipeline` without changing behaviour: the emits, the
log lines, and the order of work inside each stage are as they were, so a
graph built from these stages produces the same answers as the straight
line. The eval harness is what proves that, and it can only prove it if
there is exactly one implementation of each stage to compare.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.progress import Progress
from app.prompt import REFUSAL, build_prompt, split_citations
from app.schemas import AskRequest, AskResponse, Citation, RefusalReason
from app.scope import split_marker
from app.services.base import Generation, LLMGateway, Passage, Retriever
from app.tax_year import extract_tax_year

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Everything a stage might need, and the clock they all share.

    Passed whole so a stage signature says what the stage does rather than
    re-listing the dependencies. `started` is set once, at the top of the
    request, because `latency_ms` has to mean "since the caller asked".
    """

    request: AskRequest
    retriever: Retriever
    gateway: LLMGateway
    settings: Settings
    report: Progress
    started: float

    @classmethod
    def open(
        cls,
        request: AskRequest,
        retriever: Retriever,
        gateway: LLMGateway,
        settings: Settings,
        progress: Progress | None = None,
    ) -> Context:
        return cls(
            request=request,
            retriever=retriever,
            gateway=gateway,
            settings=settings,
            report=progress or Progress.discarded(),
            started=time.perf_counter(),
        )

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


@dataclass(frozen=True)
class YearResolution:
    tax_year: int | None
    source: str


@dataclass(frozen=True)
class Retrieval:
    passages: list[Passage]
    retrieval_ms: int
    #: True when the tax-year filter was dropped to find these. A label for
    #: logs and tests - the graph's wiring, not this flag, is what stops a
    #: widened search from widening again.
    widened: bool = False


@dataclass(frozen=True)
class Verification:
    """What the model claimed, and how much of it held up."""

    declared: str | None
    answer_text: str
    cited: list[Passage]
    unsupported: set[tuple[str, str]]

    @property
    def fabricated_only(self) -> bool:
        """Cited provisions, and every one of them was invented.

        The case worth a second attempt: the model was willing to answer
        and had the provisions in front of it, but referenced something it
        was never given.
        """
        return bool(self.unsupported) and not self.cited


def resolve_tax_year(request: AskRequest) -> YearResolution:
    """Which year's law applies, and how we know."""
    if request.tax_year is not None:
        return YearResolution(request.tax_year, "request")

    detected = extract_tax_year(request.question)

    return YearResolution(detected, "question" if detected else "none")


def governing_act(tax_year: int | None) -> str | None:
    """Which Act that year puts in charge.

    The 1961 Act governs through 2025, the 2025 Act from 2026; with no
    year, neither is excluded.
    """
    if tax_year is None:
        return None

    return "ITA-2025" if tax_year >= 2026 else "ITA-1961"


async def announce(context: Context) -> None:
    await context.report.emit("received", question_length=len(context.request.question))
    await context.report.emit("resolving")


async def report_year(context: Context, year: YearResolution) -> None:
    await context.report.emit(
        "year_resolved",
        tax_year=year.tax_year,
        source=year.source,
        governing_act=governing_act(year.tax_year),
    )

    logger.info(
        "Ask started | tax_year=%s | source=%s | top_k=%s",
        year.tax_year,
        year.source,
        context.settings.retrieval_top_k,
    )


async def retrieve(context: Context, year: YearResolution) -> Retrieval:
    """Find the provisions. The only source an answer may be built from."""
    passages = await context.retriever.search(
        question=context.request.question,
        top_k=context.settings.retrieval_top_k,
        tax_year=year.tax_year,
        progress=context.report,
    )

    return Retrieval(passages=passages, retrieval_ms=context.elapsed_ms())


async def retrieve_widened(context: Context, year: YearResolution) -> Retrieval:
    """Search again without the year filter, when the filter starved it.

    A question that names no year, or names one whose Act does not carry
    the provision, can filter away the very section that answers it. The
    year filter is the most likely thing to have been wrong, so it is the
    one thing dropped - top_k and the score floor stay put, because
    loosening those trades precision for recall and that is a different
    experiment.
    """
    await context.report.emit("widening", dropped_filter="tax_year", was=year.tax_year)

    passages = await context.retriever.search(
        question=context.request.question,
        top_k=context.settings.retrieval_top_k,
        tax_year=None,
        progress=context.report,
    )

    return Retrieval(passages=passages, retrieval_ms=context.elapsed_ms(), widened=True)


async def make_prompt(context: Context, passages: list[Passage]) -> str:
    await context.report.emit("prompting", provisions=len(passages), rules=6)

    return build_prompt(context.request.question, passages)


#: Said to the model only on a second attempt, and only about citations it
#: actually invented. Naming the exact references is the point: a general
#: "do not fabricate" reminder is already rule 2 of the system prompt, and
#: the first attempt had it.
_FABRICATION_NOTE = """\
Your previous answer cited provisions that were not supplied to you: {refs}. \
Those do not exist in the material above. Answer again using only the \
numbered provisions above, citing each exactly as its heading writes it. If \
they do not cover the question, say that instead of citing anything else."""


async def make_retry_prompt(
    context: Context, passages: list[Passage], verification: Verification
) -> str:
    """The same prompt, plus the specific fabrication to avoid repeating."""
    invented = sorted(f"{act} s.{number}" for act, number in verification.unsupported)

    await context.report.emit(
        "retrying", reason="fabricated_citations", invented=invented
    )

    logger.info("Retrying generation after fabricated citations | refs=%s", invented)

    return build_prompt(
        context.request.question,
        passages,
        note=_FABRICATION_NOTE.format(refs=", ".join(invented)),
    )


async def generate(
    context: Context, prompt: str, *, attempt: int = 1
) -> tuple[Generation, int]:
    """Hand the provisions to the model. Returns the result and its cost in ms."""
    await context.report.emit(
        "generating",
        prompt_characters=len(prompt),
        max_tokens=context.request.max_tokens,
        attempt=attempt,
    )

    generation_started = time.perf_counter()

    result = await context.gateway.generate(
        prompt=prompt, max_tokens=context.request.max_tokens
    )

    llm_ms = int((time.perf_counter() - generation_started) * 1000)

    await context.report.emit(
        "generated",
        model=result.model,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
        took_ms=llm_ms,
    )

    if result.finish_reason == "length":
        # Reasoning tokens count against max_tokens, so a model that thinks
        # hard can exhaust the budget before it finishes writing.
        logger.warning(
            "Answer truncated | completion_tokens=%s | reasoning_tokens=%s | "
            "raise max_tokens",
            result.completion_tokens,
            result.reasoning_tokens,
        )

    return result, llm_ms


async def verify(
    context: Context, result: Generation, passages: list[Passage]
) -> Verification:
    """Check every citation against what was actually supplied."""
    await context.report.emit("verifying")

    # The model's own signal, taken off the front of the answer before
    # anything else reads it. The model marks the case; this service decides
    # what that means.
    declared, answer_text = split_marker(result.text)

    cited, unsupported = split_citations(answer_text, passages)

    await context.report.emit(
        "verified",
        cited=[f"{p.act} s.{p.section_number}" for p in cited],
        invented=sorted(f"{act} s.{number}" for act, number in unsupported),
    )

    if unsupported:
        # The model cited a provision it was never given. Harmless to the
        # caller, but the first sign the prompt is losing control of it.
        logger.warning(
            "Unsupported citations in answer | count=%s | refs=%s",
            len(unsupported),
            sorted(f"{act} s.{number}" for act, number in unsupported),
        )

    return Verification(
        declared=declared,
        answer_text=answer_text,
        cited=cited,
        unsupported=unsupported,
    )


def decide_refusal(verification: Verification) -> RefusalReason:
    """Ordered by what the caller most needs to know.

    Out of scope outranks ungrounded: a question that should not have been
    answered was not answered badly, and the two call for different fixes -
    a prompt change versus a retrieval change.

    A NOT_IN_CORPUS marker was tried here too, so the model could declare
    "I read these and they do not cover it" rather than have it inferred
    from an absent citation. It scored worse: the model emitted it on
    questions it could have answered in part, refusing wholesale where
    rule 4 wanted a partial answer. Left in RefusalReason and in scope.py
    for a later attempt, but not asked for.
    """
    if verification.declared == "out_of_scope":
        return "out_of_scope"

    if verification.cited:
        return "none"

    return "not_grounded"


def corpus_date(passages: list[Passage], settings: Settings) -> str:
    # The date belongs to the data that was retrieved, not to this process,
    # so it stays true even if the service outlives the index that built it.
    for passage in passages:
        if passage.corpus_date and passage.corpus_date != "unknown":
            return passage.corpus_date

    return settings.corpus_date


async def refuse_nothing_retrieved(
    context: Context, year: YearResolution, retrieval: Retrieval
) -> AskResponse:
    """Refusal is structural: with nothing retrieved the model is never called.

    So it cannot be talked into answering from memory.
    """
    logger.info(
        "Ask refused, nothing retrieved | retrieval_ms=%s | tax_year=%s",
        retrieval.retrieval_ms,
        year.tax_year,
    )

    await context.report.emit(
        "refused",
        reason="nothing_retrieved",
        # The model is never called on this path, so it cannot be talked
        # into answering from memory.
        model_called=False,
    )

    return AskResponse(
        answer=REFUSAL,
        refused=True,
        refusal_reason="nothing_retrieved",
        citations=[],
        corpus_date=context.settings.corpus_date,
        tax_year=year.tax_year,
        tax_year_source=year.source,
        retrieved=0,
        latency_ms=context.elapsed_ms(),
        retrieval_ms=retrieval.retrieval_ms,
    )


async def finalise(
    context: Context,
    year: YearResolution,
    retrieval: Retrieval,
    result: Generation,
    verification: Verification,
    llm_ms: int,
) -> AskResponse:
    """Turn the verified answer into the response, and say what happened."""
    refusal_reason = decide_refusal(verification)
    refused = refusal_reason != "none"

    if refused:
        await context.report.emit("refused", reason=refusal_reason, model_called=True)

    latency_ms = context.elapsed_ms()

    logger.info(
        "Ask completed | refused=%s | refusal_reason=%s | retrieved=%s | cited=%s | "
        "unsupported=%s | retrieval_ms=%s | llm_ms=%s | latency_ms=%s | "
        "model=%s | total_tokens=%s | finish_reason=%s",
        refused,
        refusal_reason,
        len(retrieval.passages),
        len(verification.cited),
        len(verification.unsupported),
        retrieval.retrieval_ms,
        llm_ms,
        latency_ms,
        result.model,
        result.total_tokens,
        result.finish_reason,
    )

    return AskResponse(
        answer=verification.answer_text,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=[]
        if refusal_reason == "out_of_scope"
        else [
            Citation(
                act=p.act,
                section_number=p.section_number,
                section_title=p.section_title,
                page_start=p.page_start,
                score=p.score,
            )
            for p in verification.cited
        ],
        unsupported_citations=sorted(
            f"{act} s.{number}" for act, number in verification.unsupported
        ),
        corpus_date=corpus_date(retrieval.passages, context.settings),
        tax_year=year.tax_year,
        tax_year_source=year.source,
        retrieved=len(retrieval.passages),
        latency_ms=latency_ms,
        retrieval_ms=retrieval.retrieval_ms,
        llm_ms=llm_ms,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        total_tokens=result.total_tokens,
        finish_reason=result.finish_reason,
    )
