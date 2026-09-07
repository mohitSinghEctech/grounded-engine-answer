from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.errors import InvalidUpstreamResponse, UpstreamRateLimited, UpstreamUnavailable
from app.retry import calculate_retry_delay
from app.services.base import LLMResult
from app.services.openai_compatible import OpenAICompatibleClient


def make_settings():
    return SimpleNamespace(
        max_retries=2,
        retry_base_delay_seconds=0.5,
        retry_max_delay_seconds=5.0,
        max_retry_time_seconds=60.0,
    )


@pytest.mark.anyio
async def test_retry_succeeds_after_two_failures(monkeypatch):
    client = OpenAICompatibleClient(
        client=None,
        settings=make_settings(),
    )

    result = SimpleNamespace(
        text="success",
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=0,
        total_tokens=30,
        finish_reason="stop",
    )

    generate_once = AsyncMock(
        side_effect=[
            UpstreamUnavailable(),
            UpstreamUnavailable(),
            result,
        ]
    )

    monkeypatch.setattr(
        client,
        "_generate_once",
        generate_once,
    )

    sleep_mock = AsyncMock()

    monkeypatch.setattr(
        "app.services.openai_compatible.asyncio.sleep",
        sleep_mock,
    )

    output = await client.generate(
        prompt="What is FastAPI?",
        max_tokens=300,
    )

    assert output.text == "success"
    assert generate_once.await_count == 3
    assert sleep_mock.await_count == 2


@pytest.mark.anyio
async def test_invalid_upstream_response_is_not_retried(monkeypatch):
    client = OpenAICompatibleClient(
        client=None,
        settings=make_settings(),
    )

    generate_once = AsyncMock(side_effect=InvalidUpstreamResponse())

    monkeypatch.setattr(
        client,
        "_generate_once",
        generate_once,
    )

    sleep_mock = AsyncMock()

    monkeypatch.setattr(
        "app.services.openai_compatible.asyncio.sleep",
        sleep_mock,
    )

    with pytest.raises(InvalidUpstreamResponse):
        await client.generate(
            prompt="What is FastAPI?",
            max_tokens=300,
        )

    assert generate_once.await_count == 1
    assert sleep_mock.await_count == 0


def test_calculate_retry_delay():
    delay_0 = calculate_retry_delay(
        attempt=0,
        base_delay=0.5,
        max_delay=5.0,
    )

    delay_1 = calculate_retry_delay(
        attempt=1,
        base_delay=0.5,
        max_delay=5.0,
    )

    delay_2 = calculate_retry_delay(
        attempt=2,
        base_delay=0.5,
        max_delay=5.0,
    )

    assert delay_1 > delay_0
    assert delay_2 > delay_1

    assert delay_0 <= 0.5 * 1.25
    assert delay_1 <= 1.0 * 1.25
    assert delay_2 <= 2.0 * 1.25


def test_calculate_retry_delay_has_jitter():
    delays = [
        calculate_retry_delay(
            attempt=1,
            base_delay=1.0,
            max_delay=10.0,
        )
        for _ in range(20)
    ]

    assert len(set(delays)) > 1


@pytest.mark.anyio
async def test_retry_skips_retry_when_time_budget_exceeded(
    llm_client,
    monkeypatch,
):
    llm_client.settings.max_retry_time_seconds = 0.0

    generate_once = AsyncMock(side_effect=UpstreamUnavailable())

    monkeypatch.setattr(
        llm_client,
        "_generate_once",
        generate_once,
    )

    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        "app.services.openai_compatible.asyncio.sleep",
        sleep_mock,
    )

    with pytest.raises(UpstreamUnavailable):
        await llm_client.generate(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert generate_once.await_count == 1
    assert sleep_mock.await_count == 0


@pytest.mark.anyio
async def test_retry_uses_retry_after_value(
    llm_client,
    monkeypatch,
):
    errors = [
        UpstreamRateLimited(retry_after_seconds=2.0),
        None,
    ]

    generate_once = AsyncMock(
        side_effect=[
            errors[0],
            LLMResult(
                text="Success",
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                total_tokens=30,
                finish_reason="stop",
            ),
        ]
    )

    monkeypatch.setattr(
        llm_client,
        "_generate_once",
        generate_once,
    )

    sleep_mock = AsyncMock()

    monkeypatch.setattr(
        "app.services.openai_compatible.asyncio.sleep",
        sleep_mock,
    )

    result = await llm_client.generate(
        prompt="What is FastAPI?",
        max_tokens=100,
    )

    assert result.text == "Success"
    assert generate_once.await_count == 2
    sleep_mock.assert_awaited_once_with(2.0)
