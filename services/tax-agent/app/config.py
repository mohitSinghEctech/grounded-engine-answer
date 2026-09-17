from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Tax Agent"
    app_version: str = "0.1.0"

    environment: Literal["development", "test", "production"] = "development"

    # Browsers refuse a cross-origin request unless the server says it is
    # allowed. A page opened from a file:// URL sends Origin: null, and a
    # page on any other host sends its own, so without this a web UI cannot
    # call this service at all - the request is blocked before it is sent.
    #
    # Comma-separated list of origins, or "*" for any. Empty is the default
    # and leaves the middleware off entirely: an API with no browser client
    # has no reason to advertise itself to one.
    cors_allow_origins: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_gateway_url: str
    llm_gateway_timeout_seconds: float = 60.0

    corpus_date: str = "unknown"

    # Which implementations to plug in. Both resolve through
    # app/retrieval/, which holds one registry per socket; an unknown name
    # fails at startup rather than on the first request.
    #
    #   embedder_provider      openai
    #   vector_store_provider  qdrant, qdrant_embedded, memory
    embedder_provider: str = "openai"
    vector_store_provider: str = "qdrant"

    # Only needed by the "qdrant" store, which talks to a server.
    qdrant_url: str | None = None

    # Only needed by "qdrant_embedded", which opens an index off the local
    # filesystem with no server at all. That is what makes a single
    # self-contained container possible: no second process to reach.
    qdrant_path: str = "data/index"

    qdrant_collection: str = "ita_sections"

    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    embedding_api_key: str
    embedding_base_url: str | None = None

    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.30

    # A section is stored as several chunks - 3,286 of them over 1,423
    # sections - and without a cap one section can take the whole window.
    # Measured: s.234A took 4 of 6 slots on one query, and on SY-03 two
    # chunks of s.24 crowded out s.23 entirely, so the answer silently lost
    # a third of the law.
    #
    # Not 1, because guidance pages carry one section_number for the whole
    # page split across many chunks; collapsing to one would throw away
    # most of a page the procedural questions need.
    retrieval_max_per_section: int = 2

    # Ask Qdrant for more than top_k, since capping discards some. Too low
    # and the window cannot be filled after the cap.
    retrieval_over_fetch: int = 4

    # Which orchestrator runs a question.
    #
    #   linear  the straight line in routers/ask.py
    #   graph   the LangGraph graph in app/graph/
    #
    # Both call the same stages in app/pipeline/steps.py, so with the two
    # branch flags below off they do identical work. Kept switchable so the
    # eval harness can score one against the other instead of the change
    # being taken on trust.
    pipeline: Literal["linear", "graph"] = "linear"

    # The two conditional branches only the graph can take. Off by default:
    # each is a behaviour change that has to earn its place in its own eval
    # run, and turning both on at once would make neither attributable -
    # the lesson the NOT_IN_CORPUS attempt already paid for.
    #
    # Retrieved nothing under a tax-year filter: search again without it.
    graph_widen_on_thin_retrieval: bool = False

    # Answer cited provisions and every one was invented: ask once more,
    # naming them. Costs a second model call on the answers that trip it.
    graph_retry_on_fabrication: bool = False

    # Send comparison questions down the agent path, where the model calls
    # tools itself. Off by default: it is a behaviour change and costs
    # roughly 3x the tokens on the questions it fires for, so it has to
    # earn that in its own eval run.
    graph_agent_on_comparison: bool = False

    # The agent's two bounds. Neither depends on the model behaving: it
    # stops at whichever it reaches first.
    agent_max_steps: int = 6
    agent_token_budget: int = 40_000

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
