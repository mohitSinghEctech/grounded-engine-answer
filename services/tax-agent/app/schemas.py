from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
        description="A question about Indian income tax law",
    )

    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=8000,
        description="Maximum tokens to generate",
    )

    tax_year: int | None = Field(
        default=None,
        ge=1961,
        le=2100,
        description="Starting year of the FY. Overrides detection from the question.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the deduction limit under section 80C?",
                "max_tokens": 800,
            }
        }
    )


class Citation(BaseModel):
    act: str
    section_number: str | None
    section_title: str | None
    page_start: int | None
    score: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    corpus_date: str

    tax_year: int | None
    tax_year_source: Literal["request", "question", "none"]
    retrieved: int

    latency_ms: int

    # Absent on a refusal: no model is called, so there is nothing to report.
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class SearchRequest(BaseModel):
    question: Question = Field(description="Text to search the corpus for")

    top_k: int = Field(default=6, ge=1, le=20)

    tax_year: int | None = Field(
        default=None,
        ge=1961,
        le=2100,
        description="Starting year of the FY. Overrides detection from the question.",
    )


class PassageOut(BaseModel):
    text: str
    score: float
    act: str
    section_number: str | None
    section_title: str | None
    page_start: int | None
    corpus_date: str


class SearchResponse(BaseModel):
    query: str
    tax_year: int | None
    tax_year_source: Literal["request", "question", "none"]
    collection: str
    passages: list[PassageOut]
