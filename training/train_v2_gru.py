"""
training/train_v2_gru.py
========================

MODEL VERSION 2 — Improved GRU
------------------------------
An improved model over the V1 baseline. Improvements:
  * Stacked GRU layers with Dropout regularization.
  * Same engineered technical indicators (MA_14, RSI, MACD, Daily_Return).
  * EarlyStopping (restore best weights) to avoid overfitting.
  * ReduceLROnPlateau adaptive learning-rate schedule.
  * A small Dense head with non-linearity.

Tracked end-to-end with MLflow (params, metrics, model file, plots).

Run:
    python training/train_v2_gru.py

Academic project — NOT financial advice.
"""

from __future__ import annotations

import json
import os
import sys

# Allow `python training/train_v2_gru.py` from the project root by ensuring the
# project root is importable (so `src` and `training` packages resolve).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
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
MODEL_VERSION = "v2_gru_improved"
LOOKBACK = 60
EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT = 0.2
ARCHITECTURE = "GRU(64) -> Dropout -> Dense(32, relu) -> Dropout -> Dense(1)"
IMPROVEMENTS = "GRU gating, dropout regularization, EarlyStopping, ReduceLROnPlateau, dense head"

MODEL_OUT = os.path.join(MODELS_DIR, "btc_gru_v2.keras")


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dropout(DROPOUT),
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
        mlflow.log_params({
            "model_version": MODEL_VERSION,
            "lookback_window": LOOKBACK,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "architecture": ARCHITECTURE,
            "improvements": IMPROVEMENTS,
            "n_features": input_shape[1],
        })

        model = build_model(input_shape)
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=25,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                              min_lr=1e-5, verbose=1),
        ]

        history = model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        preds, metrics = evaluate_on_test(model, data)

        mlflow.log_metrics({
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "directional_accuracy": metrics["directional_accuracy"],
            "training_loss": float(history.history["loss"][-1]),
            "validation_loss": float(history.history["val_loss"][-1]),
        })

        model.save(MODEL_OUT)
        print(f"Saved model -> {MODEL_OUT}")

        loss_png = os.path.join(ARTIFACTS_DIR, "training_loss_v2.png")
        pred_png = os.path.join(ARTIFACTS_DIR, "prediction_vs_actual_v2.png")
        save_loss_plot(history, loss_png, "V2 GRU — Training vs Validation Loss")
        save_pred_plot(data["y_test_raw"], preds, pred_png,
                       "V2 GRU — Predicted vs Actual (Test)")

        mlflow.log_artifact(MODEL_OUT, artifact_path="model")
        mlflow.log_artifact(loss_png, artifact_path="plots")
        mlflow.log_artifact(pred_png, artifact_path="plots")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="scaler")

        with open(os.path.join(ARTIFACTS_DIR, "metrics_v2.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n=== Model V2 (GRU improved) test metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
