def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "x-request-id" in response.headers


def test_health_request_id_is_preserved(client):
    response = client.get(
        "/health",
        headers={"X-Request-ID": "test-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-123"
