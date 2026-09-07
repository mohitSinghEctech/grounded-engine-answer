from typing import Annotated, Any

from pydantic import BaseModel, Field, ConfigDict, StringConstraints

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]


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
    answer: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None
