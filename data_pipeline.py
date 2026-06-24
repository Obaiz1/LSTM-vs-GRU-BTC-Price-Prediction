import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_bitcoin_data(ticker="BTC-USD", start_date="2014-01-01", end_date=None):
    """
    Fetches historical daily Bitcoin data from yfinance.
    """
    print(f"Fetching historical data for {ticker} starting from {start_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    # Ensure multi-index columns from yfinance (if any) are flattened
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"Data fetched successfully. Row count: {len(df)}")
    return df

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) using exponential moving averages.
    """
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, span1=12, span2=26, signal_span=9):
    """
    Calculates MACD (Difference between 12-day and 26-day EMAs) and Signal line (9-day EMA of MACD).
    """
    ema12 = series.ewm(span=span1, adjust=False).mean()
    ema26 = series.ewm(span=span2, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal

def engineer_features(df):
    """
    Calculates technical indicators and selects Close price as target.
    """
    print("Engineering features (RSI, MACD, MACD Signal, 14-day Moving Average)...")
    # Reset index if Date is the index to easily manage columns
    df = df.copy()
    if 'Date' not in df.columns:
        df = df.reset_index()

    # Calculate indicators based on the 'Close' price
    close_series = df['Close']
    
    df['RSI'] = calculate_rsi(close_series)
    macd_val, macd_signal = calculate_macd(close_series)
    df['MACD'] = macd_val
    df['MACD_Signal'] = macd_signal
    df['MA_14'] = close_series.rolling(window=14).mean()
    
    # Drop rows with NaN values resulting from indicators (e.g. MA_14 needs 14 days)
    df = df.dropna().reset_index(drop=True)
    print(f"Feature engineering complete. Dataset shape: {df.shape}")
    return df

def split_scale_and_window(df, lookback=60):
    """
    Splits the data chronologically (70/15/15), scales using MinMaxScaler fitted ONLY
    on training data, and creates sliding window inputs (3D shapes).
    """
    total_len = len(df)
    
    # Chronological boundaries
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    print(f"Splitting dataset chronologically: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Define features and target
    # Features: Close, Open, High, Low, Volume, RSI, MACD, MACD_Signal, MA_14
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MA_14']
    target_col = 'Close'
    
    # Split dataframes (including lookback buffer to avoid boundary loss)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size - lookback : train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size - lookback:].copy()
    
    # Extract arrays
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_col].values.reshape(-1, 1)
    
    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df[target_col].values.reshape(-1, 1)
    
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[target_col].values.reshape(-1, 1)
    
    # Capture the exact test dates corresponding to predictions (ignoring the lookback days)
    test_dates = df.iloc[train_size + val_size:]['Date'].values
    
    # Initialize Scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scalers strictly on training data
    feature_scaler.fit(X_train_raw)
    target_scaler.fit(y_train_raw)
    
    # Transform arrays
    X_train_scaled = feature_scaler.transform(X_train_raw)
    y_train_scaled = target_scaler.transform(y_train_raw)
    
    X_val_scaled = feature_scaler.transform(X_val_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)
    
    X_test_scaled = feature_scaler.transform(X_test_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)
    
    # Create sliding windows helper
    def create_windows(X_data, y_data):
        X, y = [], []
        for i in range(lookback, len(X_data)):
            X.append(X_data[i - lookback:i])
            y.append(y_data[i, 0]) # target value at index i (corresponds to step after window)
        return np.array(X), np.array(y)
    
    # Construct 3D datasets
    X_train, y_train = create_windows(X_train_scaled, y_train_scaled)
    X_val, y_val = create_windows(X_val_scaled, y_val_scaled)
    X_test, y_test = create_windows(X_test_scaled, y_test_scaled)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    
    # Keep the raw target test values to evaluate actual vs predictions in real USD prices
    y_test_raw_unwindowed = y_test_raw[lookback:]
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "test_dates": test_dates,
        "y_test_raw": y_test_raw_unwindowed,
        "test_df_raw": df.iloc[train_size + val_size:].copy(),
        "df_raw": df,
        "train_size": train_size,
        "val_size": val_size,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols
    }

def main():
    # Load and process data
    raw_df = fetch_bitcoin_data()
    engineered_df = engineer_features(raw_df)
    processed_data = split_scale_and_window(engineered_df)
    
    # Save the processed data dictionary to a pickle file for training/dashboard use
    output_path = "processed_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Successfully preprocessed and saved pipeline results to '{output_path}'.")

if __name__ == "__main__":
    main()
