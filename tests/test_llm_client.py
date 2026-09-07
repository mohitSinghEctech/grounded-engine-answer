from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.errors import (
    InvalidUpstreamResponse,
    UpstreamRateLimited,
    UpstreamTimeout,
    UpstreamUnavailable,
)


@pytest.mark.anyio
async def test_generate_once_maps_timeout_to_upstream_timeout(
    llm_client,
    mock_client,
):
    request = httpx.Request(
        "POST",
        "http://test-server/v1/chat/completions",
    )

    mock_client.chat.completions.create.side_effect = APITimeoutError(request=request)

    with pytest.raises(UpstreamTimeout) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "UPSTREAM_TIMEOUT"
    assert exc_info.value.status_code == 504
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_maps_rate_limit_and_retry_after(
    llm_client,
    mock_client,
):
    request = httpx.Request(
        "POST",
        "http://test-server/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=429,
        headers={"retry-after": "2.0"},
        request=request,
    )

    mock_client.chat.completions.create.side_effect = RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )

    with pytest.raises(UpstreamRateLimited) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "UPSTREAM_RATE_LIMITED"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_seconds == 2.0
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_maps_connection_error_to_upstream_unavailable(
    llm_client,
    mock_client,
):
    request = httpx.Request(
        "POST",
        "http://test-server/v1/chat/completions",
    )

    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=request
    )

    with pytest.raises(UpstreamUnavailable) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "UPSTREAM_UNAVAILABLE"
    assert exc_info.value.status_code == 503
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_maps_5xx_to_upstream_unavailable(
    llm_client,
    mock_client,
):
    request = httpx.Request(
        "POST",
        "http://test-server/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=500,
        request=request,
    )

    mock_client.chat.completions.create.side_effect = APIStatusError(
        "upstream server error",
        response=response,
        body=None,
    )

    with pytest.raises(UpstreamUnavailable) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "UPSTREAM_UNAVAILABLE"
    assert exc_info.value.status_code == 503
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_maps_4xx_to_invalid_upstream_response(
    llm_client,
    mock_client,
):
    request = httpx.Request(
        "POST",
        "http://test-server/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=400,
        request=request,
    )

    mock_client.chat.completions.create.side_effect = APIStatusError(
        "bad request",
        response=response,
        body=None,
    )

    with pytest.raises(InvalidUpstreamResponse) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "INVALID_UPSTREAM_RESPONSE"
    assert exc_info.value.status_code == 502
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_rejects_empty_choices(
    llm_client,
    mock_client,
):
    response = MagicMock()
    response.choices = []

    mock_client.chat.completions.create.return_value = response

    with pytest.raises(InvalidUpstreamResponse) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "INVALID_UPSTREAM_RESPONSE"
    assert exc_info.value.status_code == 502
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_generate_once_rejects_none_content(
    llm_client,
    mock_client,
):
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content=None),
        )
    ]

    mock_client.chat.completions.create.return_value = response

    with pytest.raises(InvalidUpstreamResponse) as exc_info:
        await llm_client._generate_once(
            prompt="What is FastAPI?",
            max_tokens=100,
        )

    assert exc_info.value.error_code == "INVALID_UPSTREAM_RESPONSE"
    assert exc_info.value.status_code == 502
    assert mock_client.chat.completions.create.await_count == 1
