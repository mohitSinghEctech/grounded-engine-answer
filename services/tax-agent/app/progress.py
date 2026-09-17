"""Reporting what the pipeline is doing, while it is still doing it.

A question takes about three seconds: roughly half a second finding the law
and two and a half waiting for the model. A caller that gets nothing until
the end has no way to show a user what is happening, and three silent
seconds feels much longer than three narrated ones.

So the pipeline announces each step. Two rules keep this from becoming a
second code path that can drift from the first:

  * `Progress.discarded()` is a working emitter that throws everything away.
    The non-streaming `/ask` uses it, so there is no `if streaming:` branch
    anywhere in the pipeline - one path, always.
  * Steps carry *facts*, not sentences. `retrieved` reports which sections
    were found, not "Found 6 provisions!". Wording is the UI's business and
    changing it should never mean changing this service.

Deliberately not included: the answer text as it is written. Citations can
only be checked once the answer is complete, so streaming the prose would
mean showing a reader a fabricated section reference and correcting it
afterwards. Two answers in forty try to cite something that does not exist,
so that is a real risk, not a theoretical one. Steps are safe because they
describe the machinery, never the law.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

#: The step vocabulary, in the order a successful request produces them.
#: A UI can map these to its own labels; anything it does not recognise
#: should be ignored rather than displayed raw.
#:
#: Every step corresponds to work that actually happens - none are invented
#: to pad the sequence - and the pairs (embedding/embedded,
#: searching/searched) exist because the gap between them is where the time
#: goes, so a UI can show a spinner for exactly as long as the work runs.
#:
#:   received      the question passed validation
#:   resolving     working out which tax year applies
#:   year_resolved  the year, where it came from, and which Act governs
#:   embedding     turning the question into a vector - a network call
#:   embedded      how many dimensions, and how long it took
#:   searching     querying the vector store
#:   searched      how many raw candidates came back
#:   filtering     dropping candidates below the score floor
#:   looking_up    the question named a section; fetching it exactly
#:   looked_up     how many chunks of the named sections were found
#:   capping       limiting how much of the window one section may take
#:   retrieved     the final set: acts, sections and scores
#:   prompting     assembling the provisions and the rules
#:   widening      searching again with the tax-year filter dropped
#:   retrying      asking again, naming the citations that were invented
#:   generating    the provisions are with the model
#:   generated     model, tokens, finish reason, how long it took
#:   verifying     checking every citation against what was supplied
#:   verified      how many held up, and how many were invented
#:   refused       stopping without an answer, with the reason
#:   done          the full response follows
STEPS = (
    "received",
    "resolving",
    "year_resolved",
    "embedding",
    "embedded",
    "searching",
    "searched",
    "filtering",
    "looking_up",
    "looked_up",
    "capping",
    "retrieved",
    # Graph-only. Emitted when a conditional branch sends the run back
    # through a stage that already ran, which is the one thing the linear
    # pipeline cannot do.
    "widening",
    "retrying",
    "prompting",
    "generating",
    "generated",
    "verifying",
    "verified",
    "refused",
    "done",
)


@dataclass(frozen=True)
class Step:
    """One thing the pipeline did, and the facts about it."""

    name: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_sse(self, event: str = "status") -> str:
        """This step as one Server-Sent Events frame.

        SSE is the plain-HTTP way to push updates to a browser: a long-lived
        response, one `event:`/`data:` pair per update, a blank line between.
        Simpler than a WebSocket because it only goes one way, which is all
        this needs.
        """
        payload = json.dumps({"step": self.name, **self.detail})

        return f"event: {event}\ndata: {payload}\n\n"


class Progress:
    """Where steps go. Either a queue someone is draining, or nowhere."""

    def __init__(self, queue: asyncio.Queue[Step | None] | None):
        self._queue = queue

    @classmethod
    def discarded(cls) -> Progress:
        """An emitter that accepts every step and keeps none.

        So the pipeline never asks whether anyone is listening.
        """
        return cls(queue=None)

    @property
    def listening(self) -> bool:
        return self._queue is not None

    async def emit(self, name: str, **detail: Any) -> None:
        if self._queue is None:
            return

        await self._queue.put(Step(name=name, detail=detail))

    async def finish(self) -> None:
        """Signal that no more steps are coming."""
        if self._queue is not None:
            await self._queue.put(None)
