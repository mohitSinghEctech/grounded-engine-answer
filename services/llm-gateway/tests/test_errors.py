import pytest

from app.errors import (
    InvalidUpstreamResponse,
    UpstreamRateLimited,
    UpstreamTimeout,
    UpstreamUnavailable,
)


@pytest.mark.parametrize(
    "exception, expected_status, expected_error_code",
    [
        (
            UpstreamTimeout(),
            504,
            "UPSTREAM_TIMEOUT",
        ),
        (
            UpstreamUnavailable(),
            503,
            "UPSTREAM_UNAVAILABLE",
        ),
        (
            InvalidUpstreamResponse(),
            502,
            "INVALID_UPSTREAM_RESPONSE",
        ),
        (
            UpstreamRateLimited(),
            429,
            "UPSTREAM_RATE_LIMITED",
        ),
    ],
)
def test_upstream_errors(
    client,
    fake_llm_client,
    exception,
    expected_status,
    expected_error_code,
):
    fake_llm_client.error = exception

    response = client.post(
        "/ask",
        json={
            "question": "What is FastAPI?",
            "max_tokens": 300,
        },
    )

    assert response.status_code == expected_status

    data = response.json()

    assert data["error_code"] == expected_error_code
    assert data["request_id"]

    assert "traceback" not in response.text.lower()


def test_unexpected_exception_returns_500(
    client_no_raise,
    fake_llm_client,
):
    fake_llm_client.error = ValueError("this should never be exposed")

    response = client_no_raise.post(
        "/ask",
        json={
            "question": "What is FastAPI?",
            "max_tokens": 300,
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["error_code"] == "INTERNAL_SERVER_ERROR"
    assert data["message"] == "An unexpected error occurred"
    assert data["request_id"]

    assert "this should never be exposed" not in response.text
    assert "traceback" not in response.text.lower()
