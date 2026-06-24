# BTC Price Forecasting API — FastAPI + TensorFlow
FROM python:3.11-slim

# System deps occasionally needed by numpy/scipy/tensorflow wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project (models/ and artifacts/ included so the API can serve).
COPY . .

EXPOSE 8000

# Container healthcheck hitting the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)" || exit 1

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
