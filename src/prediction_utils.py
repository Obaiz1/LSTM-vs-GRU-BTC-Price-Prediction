"""
src/prediction_utils.py
=======================

Inference helpers shared by the FastAPI service and any scripted prediction.

Responsibilities
----------------
* Locate and lazily load the trained models (V1 LSTM / V2 GRU) and the scaler
  bundle produced by ``src/preprocessing.py``.
* Build a valid ``(1, lookback, n_features)`` model input from one of:
    1. A list of ``lookback`` feature records (provided in the request body).
    2. The latest ``lookback`` days fetched live from Yahoo Finance.
    3. A single most-recent record appended to ``lookback-1`` live history days
       (matches the simplified single-record API example).
* Run the model, inverse-transform to USD, and derive a Up/Down trend signal.

Models are loaded once and cached. TensorFlow is imported lazily so this module
can be inspected (syntax-checked) in CI without TF installed.

Academic project — NOT financial advice.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from src.preprocessing import (
    DEFAULT_LOOKBACK,
    FEATURE_COLS,
    engineer_features,
    load_btc_data,
    load_scaler_artifact,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Map a friendly version key -> (filename, human label).
MODEL_REGISTRY = {
    "v1": ("btc_lstm_v1.keras", "v1_lstm_baseline"),
    "v2": ("btc_gru_v2.keras", "v2_gru_improved"),
    # Backwards-compatible aliases to the original repo models.
    "lstm_legacy": ("lstm_model.keras", "legacy_lstm"),
    "gru_legacy": ("gru_model.keras", "legacy_gru"),
}
DEFAULT_VERSION = "v2"

_MODEL_CACHE: Dict[str, object] = {}
_SCALER_CACHE: Optional[Dict] = None


# --------------------------------------------------------------------------- #
# Path / existence helpers
# --------------------------------------------------------------------------- #
def model_path(version: str = DEFAULT_VERSION) -> str:
    """Absolute path to a registered model's .keras file."""
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model version '{version}'. "
            f"Choose from {list(MODEL_REGISTRY)}."
        )
    filename, _ = MODEL_REGISTRY[version]
    # Prefer models/ dir; fall back to project root (legacy models live there).
    candidate = os.path.join(MODELS_DIR, filename)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(_PROJECT_ROOT, filename)


def model_exists(version: str = DEFAULT_VERSION) -> bool:
    try:
        return os.path.exists(model_path(version))
    except ValueError:
        return False


def model_label(version: str = DEFAULT_VERSION) -> str:
    return MODEL_REGISTRY.get(version, (None, version))[1]


def scaler_exists() -> bool:
    from src.preprocessing import SCALER_PATH

    return os.path.exists(SCALER_PATH)


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
def get_scaler() -> Dict:
    global _SCALER_CACHE
    if _SCALER_CACHE is None:
        _SCALER_CACHE = load_scaler_artifact()
    return _SCALER_CACHE


def get_model(version: str = DEFAULT_VERSION):
    if version not in _MODEL_CACHE:
        from tensorflow.keras.models import load_model  # lazy import

        path = model_path(version)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file for version '{version}' not found at {path}. "
                f"Train it first (see training/)."
            )
        _MODEL_CACHE[version] = load_model(path)
    return _MODEL_CACHE[version]


# --------------------------------------------------------------------------- #
# Input construction
# --------------------------------------------------------------------------- #
def _records_to_matrix(records: List[Dict], feature_cols: List[str]) -> np.ndarray:
    """Convert a list of dict records into an ordered (n, n_features) matrix."""
    rows = []
    for rec in records:
        # accept both 'ma_14' style and 'MA_14' style keys.
        norm = {k.lower(): v for k, v in rec.items()}
        rows.append([float(norm[c.lower()]) for c in feature_cols])
    return np.asarray(rows, dtype="float32")


def latest_feature_window(lookback: int, feature_cols: List[str]) -> np.ndarray:
    """Fetch live BTC data and return the most recent ``lookback`` rows (raw)."""
    raw = load_btc_data()
    feats = engineer_features(raw)
    return feats[feature_cols].tail(lookback).values.astype("float32")


def build_model_input(
    sequence: Optional[List[Dict]] = None,
    single_record: Optional[Dict] = None,
    lookback: Optional[int] = None,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Produce a scaled ``(1, lookback, n_features)`` array ready for the model.

    Priority:
      1. ``sequence`` of exactly ``lookback`` records -> used directly.
      2. ``single_record`` -> appended to (lookback-1) live history days.
      3. neither -> latest ``lookback`` live days.
    """
    bundle = get_scaler()
    feature_cols = feature_cols or bundle["feature_cols"]
    lookback = lookback or bundle["lookback"]
    feature_scaler = bundle["feature_scaler"]

    if sequence:
        if len(sequence) != lookback:
            raise ValueError(
                f"'sequence' must contain exactly {lookback} records, "
                f"got {len(sequence)}."
            )
        raw_matrix = _records_to_matrix(sequence, feature_cols)
    elif single_record:
        history = latest_feature_window(lookback - 1, feature_cols)
        latest = _records_to_matrix([single_record], feature_cols)
        raw_matrix = np.vstack([history, latest])
    else:
        raw_matrix = latest_feature_window(lookback, feature_cols)

    scaled = feature_scaler.transform(raw_matrix)
    return scaled.reshape(1, lookback, len(feature_cols))


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_price(
    sequence: Optional[List[Dict]] = None,
    single_record: Optional[Dict] = None,
    version: str = DEFAULT_VERSION,
) -> Dict:
    """
    Predict the next BTC close price and a Up/Down trend signal.

    Returns a dict matching the API response contract.
    """
    bundle = get_scaler()
    target_scaler = bundle["target_scaler"]
    feature_cols = bundle["feature_cols"]
    close_idx = feature_cols.index("Close")

    X = build_model_input(sequence=sequence, single_record=single_record)
    model = get_model(version)

    pred_scaled = model.predict(X, verbose=0)
    predicted_price = float(target_scaler.inverse_transform(pred_scaled)[0, 0])

    # Last observed (scaled) close -> inverse transform for trend comparison.
    last_close_scaled = X[0, -1, close_idx]
    # Reconstruct USD close from the feature scaler (close is one feature column).
    fs = bundle["feature_scaler"]
    close_min = fs.data_min_[close_idx]
    close_max = fs.data_max_[close_idx]
    last_close = float(last_close_scaled * (close_max - close_min) + close_min)

    change = predicted_price - last_close
    pct = (change / last_close * 100.0) if last_close else 0.0
    trend = "Up" if change >= 0 else "Down"

    return {
        "predicted_price": round(predicted_price, 2),
        "last_close": round(last_close, 2),
        "change": round(change, 2),
        "pct_change": round(pct, 4),
        "trend": trend,
        "confidence_signal": _confidence_label(pct),
        "model_version": model_label(version),
        "message": "Academic prediction only, not financial advice",
    }


def _confidence_label(pct_change: float) -> str:
    """A coarse, transparent trend-strength label (NOT a probability)."""
    magnitude = abs(pct_change)
    if magnitude < 0.5:
        return "Weak / sideways trend"
    if magnitude < 2.0:
        return "Moderate trend"
    return "Strong trend"
