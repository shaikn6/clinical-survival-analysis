"""Tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers the startup event."""
    from src.api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        r = client.get("/health")
        data = r.json()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float)

    def test_health_has_models_loaded(self, client):
        r = client.get("/health")
        data = r.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], int)


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_models_status_200(self, client):
        r = client.get("/models")
        assert r.status_code == 200

    def test_models_returns_list(self, client):
        r = client.get("/models")
        data = r.json()
        assert isinstance(data, list)

    def test_model_entry_has_name(self, client):
        r = client.get("/models")
        data = r.json()
        if data:
            assert "name" in data[0]

    def test_model_entry_has_is_active(self, client):
        r = client.get("/models")
        data = r.json()
        if data:
            assert "is_active" in data[0]


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_status_200(self, client):
        payload = {
            "age": 65.0, "los": 5.0, "sofa_score": 6.0,
            "creatinine": 1.0, "lactate": 1.5, "map": 75.0, "spo2": 96.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 2.0,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_returns_survival_probs(self, client):
        payload = {
            "age": 70.0, "los": 8.0, "sofa_score": 10.0,
            "creatinine": 2.0, "lactate": 3.0, "map": 60.0, "spo2": 92.0,
            "diabetes": 1, "hypertension": 1, "heart_failure": 1, "ckd": 1, "copd": 0, "cci": 6.0,
        }
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "survival_probabilities" in data

    def test_predict_survival_probs_in_0_1(self, client):
        payload = {
            "age": 55.0, "los": 3.0, "sofa_score": 4.0,
            "creatinine": 0.8, "lactate": 1.0, "map": 80.0, "spo2": 98.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 1.0,
        }
        r = client.post("/predict", json=payload)
        data = r.json()
        probs = data.get("survival_probabilities", {})
        for key, val in probs.items():
            assert 0.0 <= val <= 1.0, f"Probability for {key} = {val} out of [0, 1]"

    def test_predict_has_risk_score(self, client):
        payload = {
            "age": 65.0, "los": 5.0, "sofa_score": 6.0,
            "creatinine": 1.0, "lactate": 1.5, "map": 75.0, "spo2": 96.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 2.0,
        }
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "risk_score" in data
        assert isinstance(data["risk_score"], float)

    def test_predict_has_model_field(self, client):
        payload = {
            "age": 65.0, "los": 5.0, "sofa_score": 6.0,
            "creatinine": 1.0, "lactate": 1.5, "map": 75.0, "spo2": 96.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 2.0,
        }
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "model" in data

    def test_predict_invalid_age_422(self, client):
        payload = {
            "age": -5.0, "los": 5.0, "sofa_score": 6.0,
            "creatinine": 1.0, "lactate": 1.5, "map": 75.0, "spo2": 96.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 2.0,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_four_time_points(self, client):
        payload = {
            "age": 65.0, "los": 5.0, "sofa_score": 6.0,
            "creatinine": 1.0, "lactate": 1.5, "map": 75.0, "spo2": 96.0,
            "diabetes": 0, "hypertension": 0, "heart_failure": 0, "ckd": 0, "copd": 0, "cci": 2.0,
        }
        r = client.post("/predict", json=payload)
        data = r.json()
        probs = data.get("survival_probabilities", {})
        assert len(probs) == 4
