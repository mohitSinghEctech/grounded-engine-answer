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

# Why an answer was withheld. "refused" alone cannot distinguish "I could
# not" from "I should not", and the two need different fixes: the first is a
# retrieval problem, the second a scope problem.
#
#   nothing_retrieved    the index returned no provision above the floor
#   not_grounded         provisions were supplied and the answer cited none
#   out_of_scope         answerable from the corpus, but asks for advice or
#                        a computed liability, which rule 3 forbids
#   not_in_corpus        the model was given provisions, read them, and said
#                        they do not cover the question - a declared refusal,
#                        as opposed to not_grounded which is only inferred
#                        from an absence of citations
#   needs_clarification  a bare section number that means different law in
#                        each Act, so the year has to be settled first
#
# The pipeline currently determines the first two structurally. The last two
# need a decision taken before retrieval and are not emitted yet.
RefusalReason = Literal[
    "none",
    "nothing_retrieved",
    "not_grounded",
    "out_of_scope",
    "not_in_corpus",
    "needs_clarification",
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

    # refused is derived from refusal_reason, never set independently, so the
    # two can never disagree.
    refused: bool
    refusal_reason: RefusalReason

    citations: list[Citation]

    # Citations the model produced for provisions it was never given. The
    # most dangerous failure this system has, because a fabricated section
    # reference looks exactly like a correct one. Returned, not merely
    # logged, so a caller and an eval can both count them.
    unsupported_citations: list[str] = []

    corpus_date: str

    tax_year: int | None
    tax_year_source: Literal["request", "question", "none"]
    retrieved: int

    latency_ms: int

    # Where the time went. retrieval_ms covers embedding the question and
    # querying Qdrant; llm_ms is the gateway round trip. They should very
    # nearly sum to latency_ms - anything left over is this service's own
    # work, and it should be small.
    retrieval_ms: int
    llm_ms: int | None = None

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
