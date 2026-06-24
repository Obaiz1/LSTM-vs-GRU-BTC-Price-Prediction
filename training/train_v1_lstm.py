"""
training/train_v1_lstm.py
=========================

MODEL VERSION 1 — Baseline LSTM
-------------------------------
A deliberately simple single-layer LSTM regression baseline. No dropout, light
tuning. Its purpose is to establish a reference point that Model V2 (improved
GRU) must beat.

Tracked end-to-end with MLflow (params, metrics, model file, plots).

Run:
    python training/train_v1_lstm.py

Academic project — NOT financial advice.
"""

from __future__ import annotations

import json
import os
import sys

# Allow `python training/train_v1_lstm.py` from the project root by ensuring the
# project root is importable (so `src` and `training` packages resolve).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

from src.preprocessing import build_dataset
from training._common import (
    ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    ensure_dirs,
    evaluate_on_test,
    save_loss_plot,
    save_pred_plot,
)

# --------------------------------------------------------------------------- #
# Hyper-parameters
# --------------------------------------------------------------------------- #
MODEL_VERSION = "v1_lstm_baseline"
LOOKBACK = 60
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ARCHITECTURE = "LSTM(64) -> Dense(1)"

MODEL_OUT = os.path.join(MODELS_DIR, "btc_lstm_v1.keras")


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(1),  # linear output for regression
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
    )
    return model


def main():
    ensure_dirs()
    data = build_dataset(lookback=LOOKBACK)
    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=MODEL_VERSION):
        # ---- params ----
        mlflow.log_params({
            "model_version": MODEL_VERSION,
            "lookback_window": LOOKBACK,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "architecture": ARCHITECTURE,
            "n_features": input_shape[1],
        })

        model = build_model(input_shape)
        model.summary()

        history = model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        preds, metrics = evaluate_on_test(model, data)

        # ---- metrics ----
        mlflow.log_metrics({
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "directional_accuracy": metrics["directional_accuracy"],
            "training_loss": float(history.history["loss"][-1]),
            "validation_loss": float(history.history["val_loss"][-1]),
        })

        # ---- save model ----
        model.save(MODEL_OUT)
        print(f"Saved model -> {MODEL_OUT}")

        # ---- plots ----
        loss_png = os.path.join(ARTIFACTS_DIR, "training_loss_v1.png")
        pred_png = os.path.join(ARTIFACTS_DIR, "prediction_vs_actual_v1.png")
        save_loss_plot(history, loss_png, "V1 LSTM — Training vs Validation Loss")
        save_pred_plot(data["y_test_raw"], preds, pred_png,
                       "V1 LSTM — Predicted vs Actual (Test)")

        # ---- log artifacts ----
        mlflow.log_artifact(MODEL_OUT, artifact_path="model")
        mlflow.log_artifact(loss_png, artifact_path="plots")
        mlflow.log_artifact(pred_png, artifact_path="plots")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # ---- persist metrics for the dashboard / comparison report ----
        with open(os.path.join(ARTIFACTS_DIR, "metrics_v1.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n=== Model V1 (LSTM baseline) test metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
