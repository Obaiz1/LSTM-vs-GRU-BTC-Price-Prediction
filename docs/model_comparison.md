# Model Comparison — V1 (LSTM) vs V2 (GRU)

**Project:** BTC Price Forecasting using LSTM and GRU with MLOps Deployment
**Category:** Finance + Time Series Forecasting + Deep Learning

> ⚠️ **Disclaimer:** This is an academic project for educational purposes only.
> It is **NOT financial advice** and must not be used to make real trading or
> investment decisions. Cryptocurrency markets are highly volatile and
> unpredictable.

---

## 1. Model Version 1 — Baseline LSTM

- **File:** `models/btc_lstm_v1.keras`
- **Script:** `training/train_v1_lstm.py`
- **Architecture:** `LSTM(64) → Dense(1)`
- **Purpose:** A deliberately simple reference model. No dropout, no callbacks,
  fixed epochs. Establishes the score that V2 must beat.
- **Inputs:** 60-day sliding window of 9 features
  (`Open, High, Low, Close, Volume, MA_14, RSI, MACD, Daily_Return`).

## 2. Model Version 2 — Improved GRU

- **File:** `models/btc_gru_v2.keras`
- **Script:** `training/train_v2_gru.py`
- **Architecture:** `GRU(96, seq) → Dropout(0.2) → GRU(48) → Dropout(0.2) → Dense(32, relu) → Dense(1)`
- **Improvements over V1:**
  - Stacked GRU layers (more representational capacity).
  - **Dropout** regularization to reduce overfitting.
  - **EarlyStopping** (restores best weights).
  - **ReduceLROnPlateau** adaptive learning-rate schedule.
  - Dense non-linear head and tuned epoch budget.
- **Inputs:** identical 60-day, 9-feature window (fair comparison).

---

## 3. Metrics Table

Metrics are computed on the held-out **test** set (chronological 70/15/15 split)
and logged to MLflow by each training script. After running both scripts the
exact numbers live in `artifacts/metrics_v1.json` and `artifacts/metrics_v2.json`.

_Example run on BTC-USD daily data (2015-01-01 → 2026-06, test split). Your
numbers will vary slightly because live data grows over time._

| Metric                 | V1 — LSTM baseline | V2 — GRU improved | Better |
|------------------------|--------------------|-------------------|--------|
| MAE (USD)              | 5,763.94           | **4,305.68**      | V2 ✅  |
| RMSE (USD)             | 6,770.52           | **5,172.28**      | V2 ✅  |
| R²                     | 0.8357             | **0.9041**        | V2 ✅  |
| Directional accuracy   | 0.4952             | 0.5064            | ~tie   |

_The dashboard's "Model Comparison" tab reads these numbers live from
`artifacts/metrics_v*.json`, so it always reflects your latest training run._

> Reference (from the original repo's `metrics.json`, single-layer LSTM vs GRU
> on the legacy feature set): LSTM RMSE ≈ **9,506**, R² ≈ **0.69**; GRU RMSE ≈
> **3,129**, R² ≈ **0.97** — i.e. the GRU clearly outperformed the LSTM
> baseline. The V1/V2 scripts re-run this comparison cleanly under MLflow.

**How to fill the table:** open the MLflow UI (`mlflow ui`), select both runs,
click **Compare**, and copy the metric values; or read the JSON files above.

---

## 4. Which model performed better?

On this dataset the **GRU (V2)** model performed clearly better on the price-level
metrics — **~25% lower MAE/RMSE and R² 0.90 vs 0.84** — because the dropout-
regularized GRU with EarlyStopping/ReduceLROnPlateau generalizes better than the
unregularized single-layer LSTM baseline. GRU's gating is also more parameter-
efficient, which helps on a modest-sized daily dataset.

Directional accuracy is ~0.50 for **both** models — i.e. neither reliably beats a
coin-flip at calling next-day up/down. This is expected and important: good RMSE
on a near-random-walk price series mostly reflects "tomorrow ≈ today", and does
**not** imply a tradeable edge. See limitations below.

## 5. Why LSTM / GRU for time-series forecasting?

- They are **recurrent** networks designed for sequential data and capture
  temporal dependencies across the 60-day window.
- **Gating mechanisms** (input/forget/output for LSTM; update/reset for GRU)
  mitigate the vanishing-gradient problem of vanilla RNNs, letting the model
  retain longer-range context.
- **GRU** uses fewer gates/parameters than LSTM, often training faster and
  generalizing well on smaller datasets — a good "improved" candidate.

## 6. Limitations of stock / crypto prediction

- Crypto markets are **near-efficient and highly volatile**; price is driven by
  news, sentiment, regulation and macro events that are **not in OHLCV data**.
- Models learn historical patterns that **may not persist** (regime change).
- Predicting the *next* close is far easier than predicting *direction* or
  *profitable* moves; good RMSE does **not** imply a profitable strategy.
- Backtest/skill metrics can be optimistic due to autocorrelation and the
  "predict tomorrow ≈ today" baseline.

## 7. Disclaimer

This project is **academic** and built to demonstrate a complete Deep Learning
+ MLOps pipeline (training, experiment tracking, serving, containerization,
CI/CD and Kubernetes). **It is not financial advice.** Do not use it for real
investment decisions.
