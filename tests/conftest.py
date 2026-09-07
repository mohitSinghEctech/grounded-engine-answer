from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_llm_client
from app.main import create_app
from app.services.base import LLMResult
from app.services.openai_compatible import OpenAICompatibleClient


class FakeLLMClient:
    def __init__(self, result=None, error=None, errors=None):
        self.result = result
        self.error = error
        self.errors = errors or []
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        self.call_count += 1

        if self.errors:
            error = self.errors.pop(0)

            if error is not None:
                raise error

        if self.error is not None:
            raise self.error

        return self.result


@pytest.fixture
def fake_llm_client():
    return FakeLLMClient(
        result=LLMResult(
            text="This is a fake answer.",
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            total_tokens=30,
            finish_reason="stop",
        )
    )


@pytest.fixture
def app(monkeypatch, fake_llm_client):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LLM_BASE_URL",
        "http://test-server/v1",
    )

    get_settings.cache_clear()

    application = create_app()

    application.dependency_overrides[get_llm_client] = lambda: fake_llm_client

    yield application

    application.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_raise(app):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def settings():
    return Settings(
        llm_api_key="test-key",
        llm_base_url="http://test-server/v1",
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def llm_client(mock_client, settings):
    return OpenAICompatibleClient(mock_client, settings)
