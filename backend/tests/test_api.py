from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_reports_model_unavailable():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_diacritization_is_not_faked():
    response = client.post("/api/diacritize", json={"text": "العلم نور"})
    assert response.status_code == 503


def test_empty_input_is_rejected():
    response = client.post("/api/diacritize", json={"text": ""})
    assert response.status_code == 422
