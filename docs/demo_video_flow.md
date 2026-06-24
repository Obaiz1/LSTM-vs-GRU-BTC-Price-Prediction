# Demo Video Flow

Suggested recording order (~5–8 minutes). Speak to each step; keep terminals and
browser windows large and readable.

> Academic project — NOT financial advice. State this on camera.

| # | Segment | What to show / say |
|---|---------|--------------------|
| 1 | **Project overview** | Title slide / README top. Goal: forecast BTC next-day close, compare LSTM vs GRU, deploy with full MLOps. |
| 2 | **Dataset description** | BTC-USD daily OHLCV from Yahoo Finance via `yfinance`; engineered indicators MA_14, RSI, MACD, Daily_Return; 60-day lookback window; 70/15/15 chronological split. |
| 3 | **LSTM Model V1** | Open `training/train_v1_lstm.py`; explain baseline `LSTM(64) → Dense(1)`. |
| 4 | **GRU Model V2** | Open `training/train_v2_gru.py`; explain stacked GRU + Dropout + EarlyStopping + ReduceLROnPlateau. |
| 5 | **Model comparison** | Show `docs/model_comparison.md` table; explain why GRU wins (lower RMSE, higher R²/directional accuracy). |
| 6 | **MLflow tracking** | `mlflow ui` → experiment with both runs → Compare view → artifacts (model + plots). |
| 7 | **FastAPI prediction** | `uvicorn deployment.api:app ...` → `/docs` → run `/predict` → show JSON (predicted_price, trend, model_version, disclaimer). |
| 8 | **React frontend** | Show the React app calling the API; enter values / use-latest, see the prediction card + trend. |
| 9 | **Docker container** | `docker build` → `docker run` → `docker ps` → `/health`. |
| 10 | **GitHub repository overview** | Walk the repo structure on GitHub. |
| 11 | **GitHub Actions CI/CD** | Actions tab → green CI run → expand steps (syntax check, YAML validate, structure tests). |
| 12 | **Kubernetes deployment** | `kubectl apply` deployment + service → `kubectl get deployments/pods/services` → `minikube service btc-price-service --url`. |
| 13 | **Final prediction demo** | Hit the live K8s/Docker endpoint once more; restate the academic disclaimer; wrap up. |
