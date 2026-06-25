# BTC Price Forecasting API — FastAPI + TensorFlow
FROM python:3.11-slim

# System deps occasionally needed by numpy/scipy/tensorflow wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project (models/ and artifacts/ included so the API can serve).
COPY . .

# Default port 8000 locally; cloud hosts (Hugging Face Spaces, Cloud Run, Render)
# inject their own $PORT, which the CMD below honors.
EXPOSE 8000

# Container healthcheck hitting the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0) if urllib.request.urlopen('http://localhost:'+p+'/health').status==200 else sys.exit(1)" || exit 1

# Shell form so ${PORT} is expanded at runtime (defaults to 8000).
CMD ["sh", "-c", "uvicorn deployment.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
