"""
src/preprocessing.py
====================

Shared preprocessing utilities for the BTC Price Forecasting final project.

This module is the SINGLE SOURCE OF TRUTH for:
  * Which features the V1 / V2 models consume (and in which order).
  * How technical indicators are computed.
  * How data is scaled, windowed and split.
  * Where the fitted scaler is persisted (``artifacts/scaler.joblib``).

It deliberately mirrors the logic in the original ``data_pipeline.py`` so the
existing Streamlit app keeps working, while exposing a clean, importable API
for the new training scripts (``training/train_v1_lstm.py`` /
``training/train_v2_gru.py``) and the FastAPI service (``deployment/api.py``).

Academic project — NOT financial advice.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Canonical feature order. BOTH model versions use exactly these features so a
# single scaler artifact can serve both, and so the FastAPI /predict input
# schema maps 1:1 to the model input.
FEATURE_COLS: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_14",
    "RSI",
    "MACD",
    "Daily_Return",
]
TARGET_COL: str = "Close"
DEFAULT_LOOKBACK: int = 60
DEFAULT_TICKER: str = "BTC-USD"
DEFAULT_START: str = "2015-01-01"

# Resolve project-root relative paths regardless of the current working dir.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def download_btc_data(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download historical daily OHLCV data from Yahoo Finance via yfinance."""
    import yfinance as yf  # imported lazily so CI / API don't require it

    print(f"[preprocessing] Downloading {ticker} from {start_date} ...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"[preprocessing] Downloaded {len(df)} rows.")
    return df


def load_btc_data(
    csv_path: str | None = None,
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load BTC data from a CSV cache if provided/available, else download it."""
    if csv_path and os.path.exists(csv_path):
        print(f"[preprocessing] Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        return df
    return download_btc_data(ticker, start_date, end_date)


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using exponential moving averages."""
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series, span1: int = 12, span2: int = 26, signal_span: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """MACD line and signal line."""
    ema12 = series.ewm(span=span1, adjust=False).mean()
    ema26 = series.ewm(span=span2, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal


def engineer_features(df: pd.DataFrame, ma_window: int = 14) -> pd.DataFrame:
    """
    Add technical indicators required by the project:
      * MA_14  (moving average)
      * RSI
      * MACD (+ MACD_Signal kept for the dashboard)
      * Daily_Return
    Missing values produced by rolling windows are dropped.
    """
    df = df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()

    close = df["Close"]
    df["MA_14"] = close.rolling(window=ma_window).mean()
    df["RSI"] = calculate_rsi(close)
    macd, macd_signal = calculate_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["Daily_Return"] = close.pct_change()

    df = df.dropna().reset_index(drop=True)
    print(f"[preprocessing] Feature engineering complete. Shape: {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# Scaling + windowing
# --------------------------------------------------------------------------- #
def _make_windows(
    X_scaled: np.ndarray, y_scaled: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(X_scaled)):
        X.append(X_scaled[i - lookback:i])
        y.append(y_scaled[i, 0])
    return np.array(X), np.array(y)


def prepare_training_data(
    df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    feature_cols: List[str] = FEATURE_COLS,
    save_scaler: bool = True,
) -> Dict:
    """
    Chronological 70/15/15 split, MinMax scaling fitted on TRAIN ONLY, then
    sliding-window construction. Returns a dictionary of train/val/test arrays
    plus the fitted scalers. Persists the scaler to ``artifacts/scaler.joblib``.
    """
    total = len(df)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size - lookback: train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size - lookback:].copy()

    feature_scaler = MinMaxScaler((0, 1))
    target_scaler = MinMaxScaler((0, 1))
    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[TARGET_COL]].values)

    def scale(part: pd.DataFrame):
        Xs = feature_scaler.transform(part[feature_cols].values)
        ys = target_scaler.transform(part[[TARGET_COL]].values)
        return Xs, ys

    Xtr, ytr = scale(train_df)
    Xva, yva = scale(val_df)
    Xte, yte = scale(test_df)

    X_train, y_train = _make_windows(Xtr, ytr, lookback)
    X_val, y_val = _make_windows(Xva, yva, lookback)
    X_test, y_test = _make_windows(Xte, yte, lookback)

    # Real-USD test targets (for inverse-transformed evaluation).
    y_test_raw = test_df[[TARGET_COL]].values[lookback:]
    test_dates = df.iloc[train_size + val_size:]["Date"].values

    if save_scaler:
        save_scaler_artifact(feature_scaler, target_scaler, feature_cols, lookback)

    print(
        f"[preprocessing] Windows -> train {X_train.shape}, "
        f"val {X_val.shape}, test {X_test.shape}"
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "y_test_raw": y_test_raw,
        "test_dates": test_dates,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "lookback": lookback,
    }


# --------------------------------------------------------------------------- #
# Scaler persistence
# --------------------------------------------------------------------------- #
def save_scaler_artifact(
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_cols: List[str],
    lookback: int,
    path: str = SCALER_PATH,
) -> str:
    """Persist scalers + metadata in a single joblib bundle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "lookback": lookback,
    }
    joblib.dump(bundle, path)
    print(f"[preprocessing] Saved scaler bundle -> {path}")
    return path


def load_scaler_artifact(path: str = SCALER_PATH) -> Dict:
    """Load the scaler bundle saved by :func:`save_scaler_artifact`."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler artifact not found at {path}. "
            f"Run a training script first (training/train_v1_lstm.py)."
        )
    return joblib.load(path)


def build_dataset(
    lookback: int = DEFAULT_LOOKBACK,
    csv_path: str | None = None,
) -> Dict:
    """Convenience one-shot: load -> engineer -> split/scale/window."""
    raw = load_btc_data(csv_path=csv_path)
    feats = engineer_features(raw)
    return prepare_training_data(feats, lookback=lookback)


if __name__ == "__main__":
    # Smoke test: build the dataset and report shapes.
    data = build_dataset()
    print("Done. Feature columns:", data["feature_cols"])
