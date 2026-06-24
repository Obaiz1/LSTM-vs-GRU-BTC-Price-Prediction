# BTC Forecasting — React Frontend

A small React + Vite single-page app that calls the FastAPI prediction service
and renders the predicted next BTC price, Up/Down trend, signal strength and the
model version used.

> Academic project — NOT financial advice.

## Run (development)

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

The dev server proxies `/api/*` to `http://localhost:8000` (the FastAPI service),
so start the API first:

```bash
# from the project root
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

## Build (production)

```bash
npm run build      # outputs to dist/
npm run preview    # serve the build locally
```

To point the build at a different API (e.g. the Minikube NodePort), set
`VITE_API_BASE` at build time:

```bash
VITE_API_BASE=http://192.168.49.2:30080 npm run build
```

## Input modes

- **Manual record** — fill the 9 feature fields; the API fetches the latest 59
  live days and appends your record as "today" to form the 60-day window.
- **Use latest live data** — tick the checkbox; the API fetches the latest 60
  days from Yahoo Finance.
- Choose **V1 (LSTM)** or **V2 (GRU)** from the model dropdown.
