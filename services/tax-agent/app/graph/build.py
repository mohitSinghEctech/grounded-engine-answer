"""Wiring the nodes into a graph, and running one question through it.

The shape, in words:

    resolve -> ? comparison question -> agent -> ? answered -> verify
             |                                  ? stopped  -> refuse_stopped
             ? otherwise
             v
               retrieve -> ? nothing found -> refuse_empty -> END
                        |                 -> widen -> ? found -> prompt
                        |                                 ? empty -> refuse_empty
                        ? found           -> prompt -> generate -> verify
                                                                     |
                             ? all citations invented -> retry_prompt -+
                             ? otherwise              -> finalise -> END

Two of those arrows are conditional, and they are the reason this is a
graph and not a function: the straight line could express "refuse when
nothing came back", but not "search again without the year filter, then
re-enter the same path" or "having seen the answer, ask once more".

Both conditional branches are off by default (`graph_widen_on_thin_
retrieval`, `graph_retry_on_fabrication`). With both off the graph walks
exactly the path the linear pipeline does, which is what makes the two
comparable in the eval harness - and each branch can then be turned on by
itself, one change per run.

No checkpointer is configured. Checkpointing is how LangGraph persists a
run so it can be paused, resumed, or replayed; nothing here needs that,
and adding it would mean every object in the state had to be
serialisable.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph import nodes
from app.graph.state import AskState
from app.pipeline.steps import Context
from app.progress import Progress
from app.schemas import AskRequest, AskResponse
from app.services.base import LLMGateway, Retriever


def build_ask_graph():
    """Define the graph. Compiled once and reused for every request."""
    builder = StateGraph(AskState)

    builder.add_node("resolve", nodes.resolve)
    builder.add_node("agent", nodes.agent)
    builder.add_node("refuse_stopped", nodes.refuse_stopped)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("widen", nodes.widen)
    builder.add_node("prompt", nodes.prompt)
    builder.add_node("retry_prompt", nodes.retry_prompt)
    builder.add_node("generate", nodes.generate)
    builder.add_node("verify", nodes.verify)
    builder.add_node("refuse_empty", nodes.refuse_empty)
    builder.add_node("finalise", nodes.finalise)

    builder.add_edge(START, "resolve")

    # The entry branch. Everything downstream of `agent` is shared with
    # the pipeline, so this is genuinely one fork and not a second
    # pipeline bolted alongside the first.
    builder.add_conditional_edges(
        "resolve",
        nodes.route_entry,
        {"pipeline": "retrieve", "agent": "agent"},
    )

    builder.add_conditional_edges(
        "agent",
        nodes.route_after_agent,
        {"verify": "verify", "refuse": "refuse_stopped"},
    )

    builder.add_conditional_edges(
        "retrieve",
        nodes.route_after_retrieval,
        {"answer": "prompt", "widen": "widen", "refuse": "refuse_empty"},
    )

    # A widened search has no filter left to drop, so its router offers
    # only answer-or-refuse. That is why there is no edge from widen back
    # to itself: the bound is in the wiring, not in a runtime check.
    builder.add_conditional_edges(
        "widen",
        nodes.route_after_widen,
        {"answer": "prompt", "refuse": "refuse_empty"},
    )

    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", "verify")

    builder.add_conditional_edges(
        "verify",
        nodes.route_after_verification,
        {"accept": "finalise", "retry": "retry_prompt"},
    )

    builder.add_edge("retry_prompt", "generate")

    builder.add_edge("refuse_empty", END)
    builder.add_edge("refuse_stopped", END)
    builder.add_edge("finalise", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_ask_graph():
    """The compiled graph, built on first use.

    Compiling validates the wiring - unreachable nodes, edges to names that
    do not exist - so it is done once at the first request rather than per
    request.
    """
    return build_ask_graph()


async def run_graph_pipeline(
    request: AskRequest,
    retriever: Retriever,
    gateway: LLMGateway,
    settings: Settings,
    progress: Progress | None = None,
) -> AskResponse:
    """Same signature as the linear `run_pipeline`, same response.

    Interchangeable on purpose: `PIPELINE=graph` swaps one for the other
    with nothing else in the service changing, so a difference in eval
    scores can only come from the orchestration.
    """
    graph = get_ask_graph()

    context = Context.open(
        request=request,
        retriever=retriever,
        gateway=gateway,
        settings=settings,
        progress=progress,
    )

    final: AskState = await graph.ainvoke({"context": context, "attempts": 0})

    return final["response"]
