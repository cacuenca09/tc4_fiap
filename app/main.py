"""
LSTM Stock Predictor — FastAPI REST API
Serve o modelo LSTM treinado para previsão de preços de ações.
"""

import os
import json
import time
import logging
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, date
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ─── Configurações de log ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Lazy import de TensorFlow ────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _load_keras_model(path: str):
    from tensorflow.keras.models import load_model
    return load_model(path)


# ─── Paths dos artefatos ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH   = os.path.join(MODEL_DIR, 'model_meta.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
MONITOR_LOG  = os.path.join(MODEL_DIR, 'monitor.jsonl')

# ─── Variáveis globais de estado ──────────────────────────────────────────────
_model  = None
_scaler = None
_meta   = {}
_request_count  = 0
_error_count    = 0
_latency_bucket: List[float] = []
APP_START = time.time()

# ─── App FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Stock Price Predictor",
    description=(
        "API RESTful para previsão de preços de fechamento de ações usando "
        "redes neurais LSTM. Treinado com dados históricos da Apple (AAPL)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        description="Lista de preços de fechamento históricos (mínimo 60 valores).",
        example=[150.0, 151.5, 152.3, 149.8],
    )

    @validator('prices')
    def validate_prices(cls, v):
        if len(v) < 60:
            raise ValueError("São necessários ao menos 60 preços históricos.")
        if any(p <= 0 for p in v):
            raise ValueError("Todos os preços devem ser positivos.")
        return v


class PredictAutoRequest(BaseModel):
    symbol: str = Field(default="AAPL", description="Ticker da ação (ex: AAPL, MSFT)")
    days_ahead: int = Field(default=5, ge=1, le=30, description="Número de dias a prever")


class PredictResponse(BaseModel):
    symbol: Optional[str]
    predictions: List[float]
    n_predictions: int
    model_meta: dict
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    request_count: int
    error_count: int
    avg_latency_ms: float
    timestamp: str


class MetricsResponse(BaseModel):
    training_metrics: dict
    model_meta: dict
    monitoring: dict


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _model, _scaler, _meta
    logger.info("🚀 Iniciando LSTM Stock Predictor API…")

    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠️  Modelo não encontrado em %s. Execute model/train.py primeiro.", MODEL_PATH)
        return

    try:
        logger.info("Carregando modelo LSTM de %s", MODEL_PATH)
        _model  = _load_keras_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _meta = json.load(f)

        logger.info("✅ Modelo carregado. SEQ_LEN=%s  Símbolo=%s",
                    _meta.get('seq_len', 60), _meta.get('symbol', 'AAPL'))
    except Exception as exc:
        logger.error("❌ Falha ao carregar modelo: %s", exc)


# ─── Middleware de monitoramento ──────────────────────────────────────────────
@app.middleware("http")
async def monitor_middleware(request: Request, call_next):
    global _request_count, _error_count, _latency_bucket
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000

    if request.url.path not in ("/health", "/metrics", "/docs", "/openapi.json", "/redoc"):
        _request_count += 1
        _latency_bucket.append(elapsed)
        if len(_latency_bucket) > 1000:
            _latency_bucket = _latency_bucket[-500:]
        if response.status_code >= 400:
            _error_count += 1

        log_entry = {
            "ts":      datetime.utcnow().isoformat(),
            "path":    request.url.path,
            "method":  request.method,
            "status":  response.status_code,
            "ms":      round(elapsed, 2),
        }
        try:
            with open(MONITOR_LOG, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass

    response.headers["X-Latency-Ms"] = f"{elapsed:.2f}"
    return response


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ensure_model():
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute model/train.py para treinar e salvar o modelo.",
        )


def _predict_from_prices(prices: List[float], days_ahead: int = 1) -> List[float]:
    seq_len    = _meta.get('seq_len', 60)
    n_features = _scaler.n_features_in_  # 10

    window = list(prices)
    results = []

    for _ in range(days_ahead):
        seq = window[-seq_len:]

        close_arr = np.array(seq, dtype=float)

        # Reconstruir as 10 features a partir só do Close
        ma7   = np.mean(close_arr[-7:])
        ma21  = np.mean(close_arr[-21:]) if len(close_arr) >= 21 else np.mean(close_arr)
        ma50  = np.mean(close_arr[-50:]) if len(close_arr) >= 50 else np.mean(close_arr)
        std21 = np.std(close_arr[-21:])  if len(close_arr) >= 21 else np.std(close_arr)
        ret   = (close_arr[-1] - close_arr[-2]) / (close_arr[-2] + 1e-9)
        hl    = 0.01   # aproximação neutra (não temos High/Low)
        oc    = 0.001  # aproximação neutra (não temos Open)

        # RSI simplificado
        diffs = np.diff(close_arr[-15:])
        gains = diffs[diffs > 0].mean() if len(diffs[diffs > 0]) > 0 else 1e-9
        losses = (-diffs[diffs < 0]).mean() if len(diffs[diffs < 0]) > 0 else 1e-9
        rsi   = 100 - (100 / (1 + gains / (losses + 1e-9)))

        volume = 50_000_000  # volume médio aproximado da AAPL

        # Montar array com as 10 features para cada timestep
        block = np.zeros((seq_len, n_features))
        for i, price in enumerate(seq):
            block[i] = [price, volume, ma7, ma21, ma50, std21, ret, hl, oc, rsi]

        scaled = _scaler.transform(block).reshape(1, seq_len, n_features)
        pred_s = _model.predict(scaled, verbose=0)

        # Inverter normalização só do Close (coluna 0)
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = pred_s[0, 0]
        pred = float(_scaler.inverse_transform(dummy)[0, 0])

        results.append(round(pred, 4))
        window.append(pred)

    return results


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "LSTM Stock Price Predictor API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health():
    avg_lat = round(sum(_latency_bucket) / len(_latency_bucket), 2) if _latency_bucket else 0.0
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        uptime_seconds=round(time.time() - APP_START, 1),
        request_count=_request_count,
        error_count=_error_count,
        avg_latency_ms=avg_lat,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoramento"])
async def get_metrics():
    """Retorna métricas de treinamento + monitoramento em produção."""
    training = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            training = json.load(f)

    recent_latencies = _latency_bucket[-100:] if _latency_bucket else [0]
    monitoring = {
        "total_requests":  _request_count,
        "total_errors":    _error_count,
        "error_rate":      round(_error_count / max(_request_count, 1) * 100, 2),
        "avg_latency_ms":  round(sum(recent_latencies) / len(recent_latencies), 2),
        "p95_latency_ms":  round(float(np.percentile(recent_latencies, 95)), 2) if len(recent_latencies) > 5 else 0,
        "uptime_seconds":  round(time.time() - APP_START, 1),
    }
    return MetricsResponse(
        training_metrics=training,
        model_meta=_meta,
        monitoring=monitoring,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predição"])
async def predict(body: PredictRequest):
    """
    Recebe uma lista de preços históricos de fechamento e retorna a previsão
    do próximo dia de negociação.

    **Requer ao menos 60 preços.**
    """
    _ensure_model()
    t0 = time.perf_counter()

    try:
        preds = _predict_from_prices(body.prices, days_ahead=1)
    except Exception as exc:
        logger.exception("Erro na previsão: %s", exc)
        raise HTTPException(status_code=500, detail=f"Erro interno na previsão: {exc}")

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(
        symbol=_meta.get('symbol'),
        predictions=preds,
        n_predictions=len(preds),
        model_meta={k: _meta.get(k) for k in ('symbol', 'trained_at', 'seq_len')},
        latency_ms=latency,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict/auto", response_model=PredictResponse, tags=["Predição"])
async def predict_auto(body: PredictAutoRequest):
    """
    Busca automaticamente os últimos 90 dias de preços via Yahoo Finance
    e retorna `days_ahead` previsões consecutivas.
    """
    _ensure_model()
    t0 = time.perf_counter()

    try:
        df = yf.download(body.symbol, period="6mo", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Símbolo '{body.symbol}' não encontrado.")

        prices = df['Close'].values.flatten().tolist()
        preds  = _predict_from_prices(prices, days_ahead=body.days_ahead)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro em /predict/auto: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(
        symbol=body.symbol,
        predictions=preds,
        n_predictions=len(preds),
        model_meta={k: _meta.get(k) for k in ('symbol', 'trained_at', 'seq_len')},
        latency_ms=latency,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info", tags=["Modelo"])
async def model_info():
    """Retorna informações sobre o modelo treinado."""
    _ensure_model()
    info = {**_meta}
    info["input_shape"]  = list(_model.input_shape) if _model else None
    info["output_shape"] = list(_model.output_shape) if _model else None
    info["n_params"]     = int(_model.count_params()) if _model else None
    return info
