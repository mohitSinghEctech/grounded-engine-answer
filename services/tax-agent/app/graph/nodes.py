"""The nodes, and the routers that choose between them.

Each node is a thin wrapper: read what it needs out of the state, call the
matching function in `app/pipeline/steps.py`, return what changed. The
work lives in the steps; the nodes exist so the graph has something to
wire together. Keeping them this thin is what makes the graph and the
straight-line pipeline produce identical answers.

The routers are the part a straight line cannot express: a plain function
that looks at the state and returns the name of the next edge.
"""

from __future__ import annotations

import logging

from app.graph.state import AskState
from app.pipeline import steps

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


async def resolve(state: AskState) -> AskState:
    """Work out which year's law applies."""
    context = state["context"]

    await steps.announce(context)

    year = steps.resolve_tax_year(context.request)

    await steps.report_year(context, year)

    return {"year": year}


async def retrieve(state: AskState) -> AskState:
    """Search the corpus. Everything downstream is built only from this."""
    retrieval = await steps.retrieve(state["context"], state["year"])

    return {"retrieval": retrieval}


async def widen(state: AskState) -> AskState:
    """Search again with the year filter dropped."""
    retrieval = await steps.retrieve_widened(state["context"], state["year"])

    return {"retrieval": retrieval}


async def prompt(state: AskState) -> AskState:
    """Assemble the provisions and the rules into one prompt."""
    text = await steps.make_prompt(state["context"], state["retrieval"].passages)

    return {"prompt": text}


async def retry_prompt(state: AskState) -> AskState:
    """Rebuild the prompt naming the citations the model invented."""
    text = await steps.make_retry_prompt(
        state["context"], state["retrieval"].passages, state["verification"]
    )

    return {"prompt": text}


async def generate(state: AskState) -> AskState:
    """Call the model."""
    attempt = state.get("attempts", 0) + 1

    result, llm_ms = await steps.generate(
        state["context"], state["prompt"], attempt=attempt
    )

    return {"result": result, "llm_ms": llm_ms, "attempts": attempt}


async def verify(state: AskState) -> AskState:
    """Check every citation against the provisions actually supplied."""
    verification = await steps.verify(
        state["context"], state["result"], state["retrieval"].passages
    )

    return {"verification": verification}


async def refuse_empty(state: AskState) -> AskState:
    """Nothing retrieved, so the model is never called. Terminal."""
    response = await steps.refuse_nothing_retrieved(
        state["context"], state["year"], state["retrieval"]
    )

    return {"response": response}


async def finalise(state: AskState) -> AskState:
    """Decide refused-or-not and build the response. Terminal."""
    response = await steps.finalise(
        state["context"],
        state["year"],
        state["retrieval"],
        state["result"],
        state["verification"],
        state["llm_ms"],
    )

    return {"response": response}


# --------------------------------------------------------------------------
# routers
# --------------------------------------------------------------------------


def route_after_retrieval(state: AskState) -> str:
    """Answer, widen the search, or refuse.

    With nothing retrieved there is nothing to answer from. Whether that
    becomes a refusal or a second, unfiltered search is the one judgment
    here, and it is settled by `graph_widen_on_thin_retrieval` rather than
    by anything the model says.
    """
    retrieval = state["retrieval"]

    if retrieval.passages:
        return "answer"

    widening_allowed = (
        state["context"].settings.graph_widen_on_thin_retrieval
        # Only worth dropping a filter that was actually applied.
        and state["year"].tax_year is not None
    )

    return "widen" if widening_allowed else "refuse"


def route_after_widen(state: AskState) -> str:
    """Answer or refuse. There is no third option left.

    A widened search has already dropped the only filter, so it cannot
    widen again. Giving the widen node its own router - rather than
    reusing the one above and relying on `retrieval.widened` to stop it -
    means the graph has no edge from widen back to itself. The bound is
    then in the wiring, where it can be read off the diagram, instead of
    in a condition someone has to trust.
    """
    return "answer" if state["retrieval"].passages else "refuse"


def route_after_verification(state: AskState) -> str:
    """Accept the answer, or give the model one more attempt.

    Only one case earns a retry: the model cited provisions and every one
    was invented. It was willing to answer and had the material in front
    of it, so the fault is in the citing, which is the thing a corrected
    prompt can actually fix. An answer that cites nothing is a retrieval
    problem and re-asking will not help; an answer with any real citation
    is already grounded.
    """
    verification = state["verification"]

    retry_allowed = (
        state["context"].settings.graph_retry_on_fabrication
        and verification.fabricated_only
        # The bound. Without it a model that fabricates every time loops
        # until something else kills the request.
        and state.get("attempts", 0) < 2
    )

    return "retry" if retry_allowed else "accept"
