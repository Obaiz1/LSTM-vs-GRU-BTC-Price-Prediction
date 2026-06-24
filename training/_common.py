"""
training/_common.py
===================

Shared helpers for both training scripts so V1 and V2 are evaluated and logged
identically (fair comparison). Keeps the per-version scripts short and readable.

Academic project — NOT financial advice.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")
MLFLOW_EXPERIMENT = "BTC_Price_Forecasting"


def ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return float(mae), rmse, float(r2)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of days where predicted direction matches actual direction."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    if len(true_dir) == 0:
        return 0.0
    return float(np.mean(true_dir == pred_dir))


def save_loss_plot(history, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="training_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="validation_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_pred_plot(y_true, y_pred, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    plt.plot(np.asarray(y_true).flatten(), label="Actual", linewidth=1.6)
    plt.plot(np.asarray(y_pred).flatten(), label="Predicted", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Test sample (chronological)")
    plt.ylabel("BTC Close (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def evaluate_on_test(model, data: Dict) -> Tuple[np.ndarray, Dict[str, float]]:
    """Predict on the test set, inverse-transform and compute all metrics."""
    target_scaler = data["target_scaler"]
    X_test = data["X_test"]
    y_test_raw = data["y_test_raw"].flatten()

    preds_scaled = model.predict(X_test, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()

    mae, rmse, r2 = regression_metrics(y_test_raw, preds)
    dir_acc = directional_accuracy(y_test_raw, preds)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "directional_accuracy": dir_acc,
    }
    return preds, metrics
