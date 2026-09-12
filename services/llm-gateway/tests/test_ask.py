def test_ask_happy_path(client, fake_llm_client):
    response = client.post(
        "/ask",
        json={
            "question": "What is FastAPI?",
            "max_tokens": 300,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "This is a fake answer."
    assert data["model"] == "fake-model"
    assert data["latency_ms"] >= 0
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 20
    assert data["reasoning_tokens"] == 0
    assert data["total_tokens"] == 30
    assert data["finish_reason"] == "stop"

    assert fake_llm_client.call_count == 1


def test_ask_empty_question_returns_422(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "max_tokens": 300,
        },
    )

    assert response.status_code == 422


def test_ask_whitespace_question_returns_422(client):
    response = client.post(
        "/ask",
        json={
            "question": "   ",
            "max_tokens": 300,
        },
    )

    assert response.status_code == 422


def test_ask_max_tokens_zero_returns_422(client):
    response = client.post(
        "/ask",
        json={
            "question": "What is FastAPI?",
            "max_tokens": 0,
        },
    )

    assert response.status_code == 422


def test_ask_max_tokens_too_large_returns_422(client):
    response = client.post(
        "/ask",
        json={
            "question": "What is FastAPI?",
            "max_tokens": 99999,
        },
    )

    assert response.status_code == 422


def test_ask_validation_error_response(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "max_tokens": 300,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error_code"] == "VALIDATION_ERROR"
    assert data["request_id"]
    assert data["details"]
    assert len(data["details"]) > 0
