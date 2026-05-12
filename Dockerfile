FROM python:3.11-slim

LABEL maintainer="lstm-stock-predictor"
LABEL description="LSTM Stock Price Predictor API"

# ─── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Workdir ──────────────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python deps ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── App source ───────────────────────────────────────────────────────────────
COPY app/    ./app/
COPY model/  ./model/

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
