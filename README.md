# BTC Price Forecasting using LSTM and GRU with MLOps Deployment

**Category:** Finance · Time Series Forecasting · Deep Learning · MLOps

A complete deep-learning final project that forecasts the next-day **Bitcoin
(BTC-USD)** closing price and compares two model versions — a baseline **LSTM
(V1)** and an improved **GRU (V2)** — with a full MLOps stack: MLflow experiment
tracking, a FastAPI prediction service, a React frontend, Docker, GitHub Actions
CI/CD, and Kubernetes (Minikube) deployment. The original Streamlit dashboard is
preserved.

> ⚠️ **Disclaimer:** This is an **academic project for educational purposes
> only**. It is **NOT financial advice**. Cryptocurrency markets are extremely
> volatile and unpredictable. Do not use this for real investment decisions.

---

## 1. Problem Statement

Can recurrent neural networks (LSTM/GRU) learn temporal patterns in historical
BTC OHLCV data and technical indicators to forecast the next closing price, and
which architecture generalizes better?

## 2. Real-World Significance

Time-series forecasting underpins demand planning, risk management and algorithmic
finance. Bitcoin is a public, high-volatility series that makes a strong teaching
case for sequence modeling, regularization, experiment tracking and the realities
(and limits) of financial prediction.

## 3. Dataset Source

- **BTC-USD** daily OHLCV from **Yahoo Finance** via the `yfinance` library.
- Engineered features: `Open, High, Low, Close, Volume, MA_14, RSI, MACD, Daily_Return`.
- 60-day lookback sliding window; chronological **70/15/15** train/val/test split;
  `MinMaxScaler` fitted on training data only.

## 4. Expected Outcomes

- Predicted next BTC close price (USD).
- Direction: **Up / Down**.
- A coarse trend-strength signal + the model version used.
- A reproducible MLflow comparison of V1 vs V2.

## 5. Project Structure

```
.
├── app.py                      # Original Streamlit dashboard (preserved)
├── data_pipeline.py            # Original data pipeline (preserved)
├── models.py                   # Original model definitions (preserved)
├── lstm_model.keras            # Original trained models (preserved)
├── gru_model.keras
├── src/
│   ├── preprocessing.py        # Shared features/scaling/windowing + scaler artifact
│   └── prediction_utils.py     # Model/scaler loading + inference helpers
├── training/
│   ├── train_v1_lstm.py        # Model V1 — baseline LSTM (MLflow tracked)
│   └── train_v2_gru.py         # Model V2 — improved GRU (MLflow tracked)
├── deployment/
│   └── api.py                  # FastAPI service: / , /health , /predict
├── frontend/                   # React + Vite single-page app
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/ci.yml    # GitHub Actions CI
├── docs/
│   ├── model_comparison.md
│   ├── commands_for_screenshots.md
│   └── demo_video_flow.md
├── tests/                      # test_api.py, test_project_structure.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-ci.txt
```

## 6. Model V1 — Baseline LSTM

`LSTM(64) → Dense(1)`. Single-layer, no regularization, fixed epochs. Establishes
the reference score. Saved to `models/btc_lstm_v1.keras`.
Train: `python training/train_v1_lstm.py`

## 7. Model V2 — Improved GRU

`GRU(96, seq) → Dropout → GRU(48) → Dropout → Dense(32, relu) → Dense(1)` with
**Dropout**, **EarlyStopping** and **ReduceLROnPlateau**. Saved to
`models/btc_gru_v2.keras`. Train: `python training/train_v2_gru.py`.
See [docs/model_comparison.md](docs/model_comparison.md) for the full comparison.

## 8. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 9. MLflow — Experiment Tracking

Both training scripts log params, metrics (MAE, RMSE, R², directional accuracy,
losses), the saved model and plots to MLflow under experiment
`BTC_Price_Forecasting`.

```bash
python training/train_v1_lstm.py
python training/train_v2_gru.py
mlflow ui --host 0.0.0.0 --port 5000   # open http://localhost:5000
```
Tick both runs → **Compare** to see V1 vs V2 side by side.

## 10. FastAPI — Model Serving

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

| Method | Path             | Description |
|--------|------------------|-------------|
| GET    | `/`              | Project metadata, endpoints, status |
| GET    | `/health`        | API status + whether model/scaler files exist |
| POST   | `/predict`       | Next-price prediction + Up/Down trend + model version |
| GET    | `/metrics`       | V1 vs V2 test metrics, run params + plain-language verdict |
| GET    | `/diagnostics`   | Training/validation loss curves + prediction-vs-actual arrays |
| GET    | `/correlation`   | 9-feature correlation matrix (heatmap) |
| GET    | `/price-history` | Recent BTC closes (forecast chart) |
| GET    | `/runs`          | MLflow-style run cards (champion + baseline) |

The analytics endpoints serve precomputed JSON from `artifacts/` (written by the
training scripts), so the dashboard works offline inside Docker / Kubernetes.

**`/predict` input modes** (sequence models need 60 timesteps):
1. `{"sequence": [ {…} × 60 ]}` — fully offline, used directly.
2. A single record `{open, high, low, close, volume, ma_14, rsi, macd, daily_return}`
   — the 59 most recent days are fetched live and your record is appended as "today".
3. `{"use_latest": true}` — latest 60 live days from Yahoo Finance.

Add `"model_version": "v1" | "v2"` (default `v2`).

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"open":65000,"high":66000,"low":64000,"close":65500,"volume":32000000000,"ma_14":64800,"rsi":58.5,"macd":120.3,"daily_return":0.012,"model_version":"v2"}'
```
```json
{ "predicted_price": 66250.75, "trend": "Up", "model_version": "v2_gru_improved",
  "message": "Academic prediction only, not financial advice" }
```

## 11. React Frontend (QuantForecaster dashboard)

A Vite + React + Tailwind + Recharts single-page analytics dashboard with four tabs:

- **Forecast** — prediction form (or live data), result card, price-history + forecast chart.
- **Model Comparison** — MAE/RMSE bars, R² gauges, feature-correlation heatmap, LSTM-vs-GRU deep-dive.
- **Training Diagnostics** — train/validation loss curves (early-stop marker) + prediction-vs-actual.
- **Experiments** — MLflow-style run cards (champion / baseline) with hyperparameters.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api -> :8000)
```

If your API is on a different port (e.g. a container on 8010), point the UI at it:

```bash
# bash
VITE_API_BASE=http://localhost:8010 npm run dev
# PowerShell
$env:VITE_API_BASE="http://localhost:8010"; npm run dev
```

Details: [frontend/README.md](frontend/README.md).

## 12. Streamlit Dashboard (original, preserved)

```bash
streamlit run app.py
```

## 13. Docker

```bash
docker build -t btc-price-final:v1 .
docker run -p 8000:8000 btc-price-final:v1
# optional full stack (API + MLflow):
docker compose up --build
```

## 14. GitHub Actions CI/CD

`.github/workflows/ci.yml` runs on push/PR to `main` and manual dispatch. It is
**deliberately lightweight** (no TensorFlow training): installs `requirements-ci.txt`,
runs `py_compile` syntax checks on all core modules, lints with ruff, verifies the
Dockerfile + Kubernetes manifests exist, validates YAML, and runs the structure
tests. See the repo's **Actions** tab for green runs.

## 15. Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)          # Windows PS: & minikube docker-env --shell powershell | Invoke-Expression
docker build -t btc-price-final:v1 .
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get deployments
kubectl get pods
kubectl get services
minikube service btc-price-service --url
```
The deployment uses `imagePullPolicy: Never` (image built into Minikube's Docker)
and `readiness`/`liveness` probes on `/health`. The service is a **NodePort** on
**30080**.

## 16. Tests

```bash
pytest tests/test_project_structure.py -q   # lightweight (CI)
pytest tests/test_api.py -q                 # full env (FastAPI + TF)
```

## 17. Required Screenshots Checklist

See [docs/commands_for_screenshots.md](docs/commands_for_screenshots.md). Summary:

- [ ] MLflow dashboard (both runs)
- [ ] MLflow run comparison
- [ ] Model artifact in MLflow
- [ ] FastAPI Swagger docs
- [ ] `/predict` JSON output (or React result card)
- [ ] Docker build success
- [ ] Running Docker container (`docker ps` + `/health`)
- [ ] GitHub Actions green run
- [ ] `kubectl get deployments`
- [ ] `kubectl get pods`
- [ ] `kubectl get services`

## 18. Demo Video Flow

See [docs/demo_video_flow.md](docs/demo_video_flow.md).

## 19. Disclaimer

This project is academic and demonstrates an end-to-end DL + MLOps pipeline.
**It is not financial advice.** Do not use it to make investment decisions.
