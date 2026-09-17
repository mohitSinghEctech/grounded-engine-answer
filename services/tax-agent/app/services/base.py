from dataclasses import dataclass
from typing import Protocol

from app.progress import Progress


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


class LLMGateway(Protocol):
    async def generate(self, prompt: str, max_tokens: int) -> Generation: ...


@dataclass(frozen=True)
class Passage:
    text: str
    score: float
    act: str
    section_number: str | None
    section_title: str | None
    page_start: int | None
    corpus_date: str


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
