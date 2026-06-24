# Commands & Screenshot Checklist

Run these in order. Each block lists the **command(s)** and the **exact screenshot**
to capture for your final-project submission. Linux/macOS shown; Windows notes added.

> Academic project — NOT financial advice.

---

## 0. Setup (once)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the models (creates models/ + artifacts/ + MLflow runs)

```bash
python training/train_v1_lstm.py
python training/train_v2_gru.py
```
📸 **Screenshot 1 — Terminal:** final printed metrics for V1 and V2.

## 2. MLflow dashboard

```bash
mlflow ui --host 0.0.0.0 --port 5000
# open http://localhost:5000
```
📸 **Screenshot 2 — MLflow dashboard:** experiment `BTC_Price_Forecasting` with
both runs (`v1_lstm_baseline`, `v2_gru_improved`) listed.

## 3. MLflow run comparison

In MLflow: tick both runs → click **Compare**.
📸 **Screenshot 3 — Compare view:** side-by-side params + metrics
(MAE, RMSE, R2, directional_accuracy).

## 4. Model artifact in MLflow

Open the V2 run → **Artifacts** tab → expand `model/`, `plots/`.
📸 **Screenshot 4 — Artifacts:** the saved `.keras` model and the
`training_loss_v2.png` / `prediction_vs_actual_v2.png` plots.

## 5. FastAPI docs (Swagger)

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs
```
📸 **Screenshot 5 — Swagger UI:** the `/`, `/health`, `/predict` endpoints.

## 6. /predict output

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"open":65000,"high":66000,"low":64000,"close":65500,"volume":32000000000,"ma_14":64800,"rsi":58.5,"macd":120.3,"daily_return":0.012,"model_version":"v2"}'
```
Windows PowerShell:
```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"open\":65000,\"high\":66000,\"low\":64000,\"close\":65500,\"volume\":32000000000,\"ma_14\":64800,\"rsi\":58.5,\"macd\":120.3,\"daily_return\":0.012,\"model_version\":\"v2\"}"
```
📸 **Screenshot 6 — JSON response:** `predicted_price`, `trend`, `model_version`,
disclaimer message. (Or capture the React app's prediction card.)

## 7. Docker build

```bash
docker build -t btc-price-final:v1 .
```
📸 **Screenshot 7 — Build output:** `Successfully tagged btc-price-final:v1`.

## 8. Docker running container

```bash
docker run -p 8000:8000 btc-price-final:v1
# in another terminal:
docker ps
curl http://localhost:8000/health
```
📸 **Screenshot 8 — `docker ps`** showing the running container + a `/health` 200.

## 9. GitHub Actions success

Push to `main`, open the repo's **Actions** tab → latest **CI** run.
📸 **Screenshot 9 — Green workflow:** all CI steps passed.

## 10. Kubernetes deployment (Minikube)

```bash
minikube start
eval $(minikube docker-env)          # Windows PS: & minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t btc-price-final:v1 .
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get deployments
```
📸 **Screenshot 10 — `kubectl get deployments`:** `btc-price-api` READY 1/1.

## 11. Kubernetes pods

```bash
kubectl get pods
```
📸 **Screenshot 11 — `kubectl get pods`:** pod `Running`.

## 12. Kubernetes services

```bash
kubectl get services
minikube service btc-price-service --url
```
📸 **Screenshot 12 — `kubectl get services`:** `btc-price-service` NodePort 30080,
plus the URL from `minikube service`.
