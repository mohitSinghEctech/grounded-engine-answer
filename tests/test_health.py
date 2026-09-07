from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "x-request-id" in response.headers
