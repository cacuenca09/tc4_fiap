"""
Testes automatizados da LSTM Stock Predictor API.
Execute com: pytest tests/test_api.py -v
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.main import app

client = TestClient(app)


# ─── Mock do modelo e scaler ──────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_model_and_scaler():
    """Injeta um modelo e scaler mock para todos os testes."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.5]])
    mock_model.input_shape  = (None, 60, 1)
    mock_model.output_shape = (None, 1)
    mock_model.count_params.return_value = 120000

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((60, 1))
    mock_scaler.inverse_transform.return_value = np.array([[175.50]])

    import app.main as main_module
    main_module._model  = mock_model
    main_module._scaler = mock_scaler
    main_module._meta   = {
        'symbol':     'AAPL',
        'seq_len':    60,
        'trained_at': '2024-01-01T00:00:00',
    }
    yield
    main_module._model  = None
    main_module._scaler = None
    main_module._meta   = {}


# ─── Testes de Saúde ──────────────────────────────────────────────────────────
class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "LSTM" in r.json()["message"]

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    def test_health_headers(self):
        r = client.get("/health")
        assert "X-Latency-Ms" in r.headers


# ─── Testes de Predição ───────────────────────────────────────────────────────
class TestPredict:
    def _prices(self, n=70):
        return [150.0 + i * 0.1 for i in range(n)]

    def test_predict_success(self):
        r = client.post("/predict", json={"prices": self._prices()})
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["n_predictions"] == 1
        assert data["latency_ms"] >= 0

    def test_predict_too_few_prices(self):
        r = client.post("/predict", json={"prices": [150.0] * 30})
        assert r.status_code == 422   # Unprocessable Entity

    def test_predict_negative_prices(self):
        prices = [-100.0] * 70
        r = client.post("/predict", json={"prices": prices})
        assert r.status_code == 422

    def test_predict_exactly_60(self):
        r = client.post("/predict", json={"prices": self._prices(60)})
        assert r.status_code == 200

    def test_predict_response_structure(self):
        r = client.post("/predict", json={"prices": self._prices()})
        data = r.json()
        required = {"symbol", "predictions", "n_predictions", "model_meta", "latency_ms", "timestamp"}
        assert required.issubset(data.keys())


# ─── Testes de Métricas ───────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_endpoint(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "monitoring" in data
        assert "total_requests" in data["monitoring"]
        assert "avg_latency_ms" in data["monitoring"]

    def test_model_info(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "AAPL"
        assert data["seq_len"] == 60
        assert data["n_params"] == 120000


# ─── Testes sem modelo carregado ──────────────────────────────────────────────
class TestNoModel:
    @pytest.fixture(autouse=True)
    def clear_model(self):
        import app.main as m
        prev_model  = m._model
        prev_scaler = m._scaler
        m._model  = None
        m._scaler = None
        yield
        m._model  = prev_model
        m._scaler = prev_scaler

    def test_predict_no_model(self):
        r = client.post("/predict", json={"prices": [150.0] * 70})
        assert r.status_code == 503

    def test_model_info_no_model(self):
        r = client.get("/model/info")
        assert r.status_code == 503
