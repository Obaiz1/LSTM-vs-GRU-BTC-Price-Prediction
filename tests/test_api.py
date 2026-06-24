"""
API tests using FastAPI's TestClient.

These require the runtime deps (fastapi, tensorflow, etc.). They are intended
to run locally / in a full environment, NOT in the lightweight CI job.

Behaviour covered:
  * GET /        -> 200
  * GET /health  -> 200 and reports artifact existence flags
  * POST /predict-> returns valid JSON if artifacts exist; otherwise a clean
                    503 error (never a crash).
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("src.prediction_utils")

from fastapi.testclient import TestClient  # noqa: E402

from deployment.api import app  # noqa: E402
from src import prediction_utils as pu  # noqa: E402

client = TestClient(app)


def test_root_ok():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "project" in body
    assert "endpoints" in body
    assert body["status"] == "ok"


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "v2_model_exists" in body
    assert "scaler_exists" in body
    assert "status" in body


def test_predict_behaviour():
    payload = {
        "open": 65000, "high": 66000, "low": 64000, "close": 65500,
        "volume": 32000000000, "ma_14": 64800, "rsi": 58.5,
        "macd": 120.3, "daily_return": 0.012,
        "model_version": "v2",
    }
    r = client.post("/predict", json=payload)

    artifacts_ready = pu.scaler_exists() and pu.model_exists("v2")
    if artifacts_ready:
        # Note: with a single record this hits live yfinance; allow 200 or a
        # clean upstream/network error (502), but never a 500 crash.
        assert r.status_code in (200, 502)
        if r.status_code == 200:
            body = r.json()
            assert "predicted_price" in body
            assert body["trend"] in ("Up", "Down")
            assert "not financial advice" in body["message"].lower()
    else:
        # Missing artifacts -> graceful 503, not a crash.
        assert r.status_code == 503
        assert "detail" in r.json()


def test_predict_missing_model_is_graceful():
    # Unknown/unavailable version should produce a handled error, never a 500.
    r = client.post("/predict", json={"use_latest": True, "model_version": "v1"})
    assert r.status_code in (200, 502, 503)
