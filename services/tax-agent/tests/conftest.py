"""Fakes for the whole pipeline, so the router can be tested offline.

Nothing here touches Qdrant, OpenAI or Docker. The retriever is the real
`GroundedRetriever` over an `InMemoryStore`, and only the LLM gateway is a
stub - which means these tests exercise the actual retrieval policy and the
actual citation checking, not a mock of them.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_llm_gateway, get_retriever
from app.main import create_app
from app.retrieval.retriever import GroundedRetriever
from app.retrieval.stores import InMemoryStore
from app.services.base import Generation


class StubEmbedder:
    """Counts marker words. Deterministic, no network."""

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "stub"

    async def embed(self, text: str) -> list[float]:
        lowered = text.lower()

        return [
            float(lowered.count("insurance")),
            float(lowered.count("interest")),
            float(lowered.count("return")),
        ]


class StubGateway:
    """Returns whatever answer the test asked for."""

    def __init__(self, text: str = "Health insurance premium (ITA-1961 s.80D)."):
        self.text = text
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int) -> Generation:
        self.calls += 1

        return Generation(
            text=self.text,
            model="stub-model",
            prompt_tokens=100,
            completion_tokens=20,
            reasoning_tokens=0,
            total_tokens=120,
            finish_reason="stop",
        )


def retrieval_settings(**overrides):
    base = dict(
        retrieval_top_k=6,
        retrieval_min_score=0.30,
        retrieval_max_per_section=2,
        retrieval_over_fetch=4,
        corpus_date="2026-09-14",
    )
    base.update(overrides)

    return SimpleNamespace(**base)


@pytest.fixture
def store():
    """Two provisions: one about insurance, one about interest."""
    memory = InMemoryStore()
    memory.add(
        [1.0, 0.0, 0.0],
        {
            "act": "ITA-1961",
            "section_number": "80D",
            "section_title": "Deduction in respect of health insurance premia",
            "text": "insurance text",
            "corpus_date": "2026-09-14",
        },
    )
    memory.add(
        [0.0, 1.0, 0.0],
        {
            "act": "ITA-1961",
            "section_number": "234A",
            "section_title": "Interest for defaults in furnishing return of income",
            "text": "interest text",
            "corpus_date": "2026-09-14",
        },
    )

    return memory


@pytest.fixture
def gateway():
    return StubGateway()


@pytest.fixture
def app(monkeypatch, store, gateway):
    # create_app builds a real lifespan, so give Settings what it demands
    # and then override the two dependencies the router actually uses.
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://gateway.invalid")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")

    get_settings.cache_clear()

    retriever = GroundedRetriever(
        embedder=StubEmbedder(),
        store=store,
        settings=retrieval_settings(),
    )

    application = create_app()
    application.dependency_overrides[get_retriever] = lambda: retriever
    application.dependency_overrides[get_llm_gateway] = lambda: gateway
    application.dependency_overrides[get_settings] = lambda: Settings(
        llm_gateway_url="http://gateway.invalid",
        qdrant_url="http://qdrant.invalid",
        embedding_api_key="test-key",
    )

    yield application

    application.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def cors_client(monkeypatch, store, gateway):
    """The same app, but with CORS switched on for any origin.

    "*" rather than a named host because a page opened from a file:// URL
    sends `Origin: null`, which no explicit allow-list entry matches.
    """
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://gateway.invalid")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    # Without this the lifespan tries to reach a real Qdrant and blocks.
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")

    get_settings.cache_clear()

    application = create_app()
    application.dependency_overrides[get_retriever] = lambda: GroundedRetriever(
        embedder=StubEmbedder(),
        store=store,
        settings=retrieval_settings(),
    )
    application.dependency_overrides[get_llm_gateway] = lambda: gateway

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
    get_settings.cache_clear()
