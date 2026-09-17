from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A function the model wants run. It cannot run anything itself.

    `arguments` stays a raw JSON string on purpose: the gateway does not
    interpret model output, it only carries it. Whoever owns the function
    parses and validates the arguments, because only they know the shape.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str

    #: Empty on an ordinary answer. Non-empty means the model answered with
    #: "run these for me" instead of with prose - so `text` will be blank
    #: and the caller is expected to execute and ask again.
    tool_calls: tuple[ToolCall, ...] = field(default=())


class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str | None = None,
        max_tokens: int = 2000,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResult:
        """One model call.

        `prompt` is the single-turn convenience every existing caller uses.
        `messages` is the real shape underneath - a conversation, which is
        what a tool-calling loop has to send because the tool results are
        part of it. Give one or the other.
        """
        ...
