from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]


Prompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200_000,
    ),
]


class ToolCallOut(BaseModel):
    """A function the model asked for. Arguments stay a raw JSON string.

    The gateway does not parse them: it has no idea what shape the caller's
    function takes, and guessing would turn a caller-side validation error
    into a gateway-side 500.
    """

    id: str
    name: str
    arguments: str


class GenerateRequest(BaseModel):
    prompt: Prompt | None = Field(
        default=None,
        description="A fully assembled prompt, sent to the model unchanged",
    )

    # The conversation form. A tool loop must send this rather than a
    # prompt, because the tool results are turns in the conversation.
    # Left as loose dicts deliberately - this is a pass-through to the
    # provider, and re-typing their message schema here would mean
    # rejecting anything they add before we catch up.
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=200,
        description="Full conversation. Use instead of prompt for a tool loop.",
    )

    tools: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=32,
        description="JSON Schema tool definitions the model may call",
    )

    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=8000,
        description="Maximum tokens to generate",
    )

    @model_validator(mode="after")
    def one_input(self):
        # Exactly one, so there is never a question about which the model saw.
        if bool(self.prompt) == bool(self.messages):
            raise ValueError("Give either prompt or messages, not both or neither.")

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Answer only from the provisions below...",
                "max_tokens": 500,
            }
        }
    )


class AskRequest(BaseModel):
    question: Question = Field(
        description="The user's question",
    )

    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=8000,
        description="Maximum tokens to generate",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is FastAPI in one sentence",
                "max_tokens": 300,
            }
        }
    )


class AskResponse(BaseModel):
    #: Empty string when the model asked for tools instead of answering.
    answer: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str

    #: Non-empty means: run these, then ask again with the results appended.
    tool_calls: list[ToolCallOut] = []


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None
