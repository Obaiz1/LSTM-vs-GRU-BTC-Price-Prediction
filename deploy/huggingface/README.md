---
title: BTC Forecasting API (LSTM vs GRU)
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# BTC Price Forecasting API

FastAPI backend for the **BTC Price Forecasting (LSTM vs GRU)** final project.
Serves the trained models + dashboard analytics endpoints.

> Academic project — **not financial advice**.

## Endpoints
- `GET /` · `GET /health` · `GET /docs`
- `POST /predict` — next-price prediction + Up/Down trend
- `GET /metrics` · `/diagnostics` · `/correlation` · `/price-history` · `/runs`

This Space runs the repository's `Dockerfile` (TensorFlow + FastAPI) on port 8000.
The React dashboard (deployed separately on Vercel) calls this Space via
`VITE_API_BASE`.

**This file is the Hugging Face Space manifest** — when deploying, it is uploaded
as the Space's `README.md` (the YAML header configures the Docker SDK + port).
