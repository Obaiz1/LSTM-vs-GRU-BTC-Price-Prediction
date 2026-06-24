"""
training/_common.py
===================

Shared helpers for both training scripts so V1 and V2 are evaluated and logged
identically (fair comparison). Keeps the per-version scripts short and readable.

Academic project — NOT financial advice.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Dashboard analytics export (consumed by the FastAPI analytics endpoints)
# --------------------------------------------------------------------------- #
def _to_float_list(arr) -> List[float]:
    return [float(x) for x in np.asarray(arr).flatten()]


def _downsample(values: List, max_points: int = 240) -> List:
    """Evenly thin a list to at most ``max_points`` so charts stay light."""
    n = len(values)
    if n <= max_points:
        return values
    step = n / max_points
    return [values[min(int(i * step), n - 1)] for i in range(max_points)]


def export_run_analytics(
    version_key: str,
    run_name: str,
    params: Dict,
    metrics: Dict,
    history,
    y_true,
    preds,
    test_dates=None,
    train_time_sec: Optional[float] = None,
) -> None:
    """Persist per-run JSON (loss curve, eval arrays, run metadata) for the UI."""
    ensure_dirs()
    hist = getattr(history, "history", {}) or {}
    loss = [float(x) for x in hist.get("loss", [])]
    val_loss = [float(x) for x in hist.get("val_loss", [])]
    epochs = len(loss)
    early_stop_epoch = (val_loss.index(min(val_loss)) + 1) if val_loss else epochs

    with open(os.path.join(ARTIFACTS_DIR, f"history_{version_key}.json"), "w") as f:
        json.dump({
            "loss": loss, "val_loss": val_loss,
            "epochs": epochs, "early_stop_epoch": early_stop_epoch,
        }, f)

    y_true_l = _to_float_list(y_true)
    preds_l = _to_float_list(preds)
    dates_l = (
        [str(d)[:10] for d in np.asarray(test_dates).flatten()]
        if test_dates is not None else []
    )
    with open(os.path.join(ARTIFACTS_DIR, f"eval_{version_key}.json"), "w") as f:
        json.dump({
            "y_true": _downsample(y_true_l),
            "y_pred": _downsample(preds_l),
            "dates": _downsample(dates_l) if dates_l else [],
            "n_test": len(y_true_l),
        }, f)

    with open(os.path.join(ARTIFACTS_DIR, f"run_{version_key}.json"), "w") as f:
        json.dump({
            "version_key": version_key,
            "name": run_name,
            "params": params,
            "metrics": metrics,
            "val_loss": (val_loss[early_stop_epoch - 1] if val_loss else None),
            "train_time_sec": round(train_time_sec, 1) if train_time_sec else None,
        }, f, indent=2)
    print(f"[analytics] Exported history/eval/run JSON for {version_key}.")


def export_shared_data() -> None:
    """Persist model-independent analytics: feature correlation + recent prices."""
    ensure_dirs()
    from src.preprocessing import FEATURE_COLS, engineer_features, load_btc_data
    try:
        feats = engineer_features(load_btc_data())
        corr = feats[FEATURE_COLS].corr().round(4)
        with open(os.path.join(ARTIFACTS_DIR, "correlation.json"), "w") as f:
            json.dump({"labels": list(FEATURE_COLS),
                       "matrix": corr.values.tolist()}, f)
        tail = feats.tail(120)
        with open(os.path.join(ARTIFACTS_DIR, "price_history.json"), "w") as f:
            json.dump({"dates": [str(d)[:10] for d in tail["Date"].values],
                       "close": [float(x) for x in tail["Close"].values]}, f)
        print("[analytics] Exported correlation + price_history JSON.")
    except Exception as exc:  # network failure must not break training
        print(f"[analytics] Skipped shared data export: {exc}")
