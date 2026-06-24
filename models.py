import os
import pickle
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_preprocessed_data(file_path="processed_data.pkl"):
    """
    Loads preprocessed data dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}. Run data_pipeline.py first.")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print("Preprocessed data loaded successfully.")
    return data

def build_lstm_model(input_shape):
    """
    Builds the LSTM sequential model with regularizations.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1) # Linear output for regression
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_gru_model(input_shape):
    """
    Builds the GRU sequential model with comparable parameter scale.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1) # Linear output for regression
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_unregularized_lstm_model(input_shape):
    """
    Builds an LSTM model WITHOUT dropout layers to serve as an unregularized baseline.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name="model", epochs=100, batch_size=32):
    """
    Trains the deep learning model with EarlyStopping and ReduceLROnPlateau callbacks.
    """
    print(f"\n--- Training {model_name.upper()} Model ---")
    
    # Overfitting Mitigation Strategy
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_reduction],
        verbose=1
    )
    
    # Save the model
    model_file = f"{model_name}_model.keras"
    model.save(model_file)
    print(f"Saved {model_name} model to {model_file}")
    
    # Save history
    history_file = f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history.history, f)
    print(f"Saved training history to {history_file}")
    
    return history

def evaluate_predictions(y_true, y_pred):
    """
    Calculates MAE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    # Load data
    data = load_preprocessed_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test_raw = data["y_test_raw"]
    target_scaler = data["target_scaler"]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input Sequence Shape: {input_shape}")
    
    # 1. Build & Train LSTM
    lstm_model = build_lstm_model(input_shape)
    lstm_model.summary()
    train_model(lstm_model, X_train, y_train, X_val, y_val, model_name="lstm", epochs=100)
    
    # 2. Build & Train GRU
    gru_model = build_gru_model(input_shape)
    gru_model.summary()
    train_model(gru_model, X_train, y_train, X_val, y_val, model_name="gru", epochs=100)
    
    # 3. Build & Train Unregularized LSTM (Baseline for comparison)
    print("\n--- Training UNREGULARIZED LSTM Model (Baseline) ---")
    unreg_lstm_model = build_unregularized_lstm_model(input_shape)
    unreg_history = unreg_lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, # 30 epochs is enough to show severe overfitting
        batch_size=32,
        verbose=1
    )
    unreg_history_file = "lstm_unreg_history.json"
    with open(unreg_history_file, "w") as f:
        json.dump(unreg_history.history, f)
    print(f"Saved unregularized LSTM history to {unreg_history_file}")

    
    # 3. Predict & Evaluate on Test Set
    print("\n--- Running Evaluations on Test Dataset ---")
    
    # Predict and inverse-transform predictions to actual USD values
    lstm_preds_scaled = lstm_model.predict(X_test)
    gru_preds_scaled = gru_model.predict(X_test)
    
    lstm_preds = target_scaler.inverse_transform(lstm_preds_scaled)
    gru_preds = target_scaler.inverse_transform(gru_preds_scaled)
    
    # Calculate metrics
    lstm_mae, lstm_rmse, lstm_r2 = evaluate_predictions(y_test_raw, lstm_preds)
    gru_mae, gru_rmse, gru_r2 = evaluate_predictions(y_test_raw, gru_preds)
    
    metrics = {
        "lstm": {"mae": float(lstm_mae), "rmse": float(lstm_rmse), "r2": float(lstm_r2)},
        "gru": {"mae": float(gru_mae), "rmse": float(gru_rmse), "r2": float(gru_r2)}
    }
    
    metrics_file = "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to {metrics_file}")
    
    # Also save predictions for plotting convenience on the dashboard
    predictions_data = {
        "lstm_preds": lstm_preds.flatten().tolist(),
        "gru_preds": gru_preds.flatten().tolist()
    }
    preds_file = "test_predictions.json"
    with open(preds_file, "w") as f:
        json.dump(predictions_data, f)
    print(f"Saved predictions to {preds_file}")
    
    print("\n--- Test Metrics Summary ---")
    print(f"LSTM: MAE={lstm_mae:.2f}, RMSE={lstm_rmse:.2f}, R2={lstm_r2:.4f}")
    print(f"GRU:  MAE={gru_mae:.2f}, RMSE={gru_rmse:.2f}, R2={gru_r2:.4f}")

if __name__ == "__main__":
    main()
