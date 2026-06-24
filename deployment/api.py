"""
deployment/api.py
=================

FastAPI service that exposes the trained BTC forecasting models.

Endpoints
---------
GET  /         -> project metadata + available endpoints + status
GET  /health   -> API status + whether model / scaler files exist
POST /predict  -> next-price prediction + Up/Down trend + model version

Sequence handling
-----------------
The models are sequence models (lookback x n_features). The /predict endpoint
accepts THREE input shapes (most reliable first):

  1. {"sequence": [ {<features>} x 60 ]}   -> used directly (fully offline).
  2. {<features>}  (a single record)       -> the 59 most recent days are
       fetched live from Yahoo Finance and your record is appended as "today".
  3. {"use_latest": true}                  -> the latest 60 live days are used.

Default model = V2 (improved GRU). Override with "model_version": "v1" | "v2".

Run:
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000

Academic project — NOT financial advice.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import prediction_utils as pu

PROJECT_NAME = "BTC Price Forecasting using LSTM and GRU with MLOps Deployment"
DISCLAIMER = "Academic prediction only, not financial advice"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


def _read_artifact(filename: str):
    """Load a JSON analytics artifact, or return None if it is not present yet."""
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

app = FastAPI(
    title=PROJECT_NAME,
    description="Compare LSTM (V1) and GRU (V2) models for Bitcoin price forecasting.",
    version="1.0.0",
)

# Allow the React frontend (and Swagger) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class FeatureRecord(BaseModel):
    open: float = Field(..., examples=[65000])
    high: float = Field(..., examples=[66000])
    low: float = Field(..., examples=[64000])
    close: float = Field(..., examples=[65500])
    volume: float = Field(..., examples=[32000000000])
    ma_14: float = Field(..., examples=[64800])
    rsi: float = Field(..., examples=[58.5])
    macd: float = Field(..., examples=[120.3])
    daily_return: float = Field(..., examples=[0.012])


class PredictRequest(BaseModel):
    # Mode 1: single record (top-level fields). All optional so other modes work.
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    ma_14: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    daily_return: Optional[float] = None

    # Mode 2: full 60-row sequence.
    sequence: Optional[List[FeatureRecord]] = None

    # Mode 3: fetch latest live data.
    use_latest: bool = False

    # Which trained model to use.
    model_version: str = pu.DEFAULT_VERSION


class PredictResponse(BaseModel):
    predicted_price: float
    last_close: float
    change: float
    pct_change: float
    trend: str
    confidence_signal: str
    model_version: str
    message: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "project": PROJECT_NAME,
        "model_type": "LSTM (V1 baseline) and GRU (V2 improved) — sequence regressors",
        "default_model_version": pu.model_label(pu.DEFAULT_VERSION),
        "endpoints": {
            "GET /": "project metadata",
            "GET /health": "service + artifact health",
            "POST /predict": "next-price prediction + trend",
            "GET /metrics": "V1 vs V2 metrics, run params + verdict",
            "GET /diagnostics": "training loss curves + prediction-vs-actual arrays",
            "GET /correlation": "feature correlation matrix (heatmap)",
            "GET /price-history": "recent BTC close prices (forecast chart)",
            "GET /docs": "interactive Swagger UI",
        },
        "status": "ok",
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "v1_model_exists": pu.model_exists("v1"),
        "v2_model_exists": pu.model_exists("v2"),
        "scaler_exists": pu.scaler_exists(),
        "default_model_version": pu.DEFAULT_VERSION,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    version = req.model_version or pu.DEFAULT_VERSION

    # Validate that required artifacts exist before doing any heavy work.
    if not pu.scaler_exists():
        raise HTTPException(
            status_code=503,
            detail="Scaler artifact missing. Train a model first "
                   "(python training/train_v1_lstm.py).",
        )
    if not pu.model_exists(version):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{version}' not found. Train it first "
                   f"(see training/). Available: v1, v2.",
        )

    # Decide the input mode.
    sequence = [r.model_dump() for r in req.sequence] if req.sequence else None

    single_record = None
    if sequence is None and not req.use_latest:
        single_fields = {
            "open": req.open, "high": req.high, "low": req.low,
            "close": req.close, "volume": req.volume, "ma_14": req.ma_14,
            "rsi": req.rsi, "macd": req.macd, "daily_return": req.daily_return,
        }
        if all(v is not None for v in single_fields.values()):
            single_record = single_fields
        # else: fall through to "use latest live data".

    try:
        result = pu.predict_price(
            sequence=sequence,
            single_record=single_record,
            version=version,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # network / yfinance / model errors
        raise HTTPException(
            status_code=502,
            detail=f"Prediction failed: {exc}. If you are offline, send a "
                   f"'sequence' of 60 records instead of relying on live data.",
        ) from exc

    return result


# --------------------------------------------------------------------------- #
# Analytics endpoints (serve precomputed JSON artifacts for the dashboard)
# --------------------------------------------------------------------------- #
@app.get("/metrics")
def metrics():
    """V1 vs V2 test metrics + run metadata + a plain-language verdict."""
    run_v1 = _read_artifact("run_v1.json")
    run_v2 = _read_artifact("run_v2.json")
    # Fall back to the bare metrics files if full run JSON isn't present.
    m1 = (run_v1 or {}).get("metrics") or _read_artifact("metrics_v1.json")
    m2 = (run_v2 or {}).get("metrics") or _read_artifact("metrics_v2.json")

    verdict = None
    if m1 and m2:
        better = "V2 GRU" if m2.get("R2", 0) >= m1.get("R2", 0) else "V1 LSTM"
        verdict = (
            f"{better} generalizes better — R² "
            f"{m2.get('R2', 0):.2f} vs {m1.get('R2', 0):.2f}. "
            f"Directional accuracy is ~0.50 for both (no tradeable edge)."
        )
    return {
        "v1": {"metrics": m1, "run": run_v1},
        "v2": {"metrics": m2, "run": run_v2},
        "verdict": verdict,
        "disclaimer": DISCLAIMER,
    }


@app.get("/diagnostics")
def diagnostics():
    """Training/validation loss curves and prediction-vs-actual test arrays."""
    return {
        "v1": {
            "history": _read_artifact("history_v1.json"),
            "eval": _read_artifact("eval_v1.json"),
        },
        "v2": {
            "history": _read_artifact("history_v2.json"),
            "eval": _read_artifact("eval_v2.json"),
        },
    }


@app.get("/correlation")
def correlation():
    """Feature correlation matrix for the heatmap (or 503 if not generated)."""
    data = _read_artifact("correlation.json")
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="correlation.json not found. Train a model to generate it.",
        )
    return data


@app.get("/price-history")
def price_history():
    """Recent BTC close prices used by the forecast chart."""
    data = _read_artifact("price_history.json")
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="price_history.json not found. Train a model to generate it.",
        )
    return data


@app.get("/runs")
def runs():
    """MLflow-style run cards (champion first), built from saved run JSON."""
    items = [r for r in (_read_artifact("run_v2.json"),
                         _read_artifact("run_v1.json")) if r]
    return {"runs": items}
