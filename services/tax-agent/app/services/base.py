from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.progress import Progress


@dataclass(frozen=True)
class ToolCall:
    """A function the model asked for. Mirrors the gateway's shape.

    `arguments` is the JSON string the model wrote. Parsing it is this
    service's job, because this service owns the functions - and it must
    parse defensively: those arguments are model output, not input the
    type system has checked.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    # Thinking tokens. They count against max_tokens but are not part of the
    # answer, so a small completion_tokens with a large total means the model
    # spent its budget reasoning and ran out of room to write.
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str

    #: Non-empty means the model wants functions run rather than having
    #: answered. `text` is then empty.
    tool_calls: tuple[ToolCall, ...] = field(default=())


class LLMGateway(Protocol):
    async def generate(
        self,
        prompt: str | None = None,
        max_tokens: int = 2000,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] = (),
    ) -> Generation:
        """One model call.

        `prompt` for the single-shot pipeline, `messages` for a tool loop -
        which must send the whole conversation, because the tool results
        are turns in it.
        """
        ...


@dataclass(frozen=True)
class Passage:
    text: str
    score: float
    act: str
    section_number: str | None
    section_title: str | None
    page_start: int | None
    corpus_date: str

    #: The counterpart section in the 1961 Act, where the corpus knows it.
    #: Populated for only a handful of sections today - which is exactly
    #: why SM-01 and SM-02 fail, and why map_section can return nothing.
    maps_to: str | None = None


class Retriever(Protocol):
    """Finds passages relevant to a question.

    Implementations must use async clients (AsyncQdrantClient, AsyncOpenAI).
    A synchronous HTTP call here blocks the whole event loop for the duration
    of the embedding round-trip, so concurrent requests queue behind it.
    """

    async def search(
        self,
        question: str,
        top_k: int,
        tax_year: int | None = None,
        progress: Progress | None = None,
    ) -> list[Passage]:
        """Find passages, reporting each phase to `progress` if given.

        The phases are part of the interface because the gap between
        embedding and searching is where the time goes, and a caller that
        wants to show a user what is happening needs to know which one it
        is waiting on.
        """
        ...
