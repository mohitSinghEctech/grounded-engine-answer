"""What the graph carries from node to node.

A LangGraph node is a function that reads this dictionary and returns the
part of it that changed; the framework merges the return value in. So the
state is the only way two nodes communicate, and reading this file tells
you everything the pipeline knows at any point.

`total=False` because the state fills up as the run proceeds: at the start
only `context` exists, and `verification` does not appear until the model
has answered. Nodes must therefore use `state.get(...)` for anything a
node before them might not have set.
"""

from __future__ import annotations

from typing import TypedDict

from app.pipeline.steps import Context, Retrieval, Verification, YearResolution
from app.schemas import AskResponse
from app.services.base import Generation


class AskState(TypedDict, total=False):
    """One question's journey through the graph."""

    #: Request, dependencies, settings, progress emitter, and the clock.
    #: Set once before the run and never replaced.
    context: Context

    year: YearResolution
    retrieval: Retrieval
    prompt: str
    result: Generation
    llm_ms: int
    verification: Verification

    #: How many times the model has been called. The retry branch reads it
    #: to guarantee it can only fire once - a loop that can re-enter itself
    #: needs a bound that does not depend on the model behaving.
    attempts: int

    #: Set by exactly one terminal node. Whatever is here when the graph
    #: stops is what the caller gets.
    response: AskResponse
