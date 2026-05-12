# 📈 LSTM Stock Price Predictor

> Modelo preditivo de redes neurais **LSTM** para previsão do preço de fechamento de ações — com pipeline completo de treinamento, avaliação e deploy via **FastAPI**.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração](#configuração)
- [Pipeline de Treinamento](#pipeline-de-treinamento)
- [API REST](#api-rest)
- [Deploy com Docker](#deploy-com-docker)
- [Monitoramento](#monitoramento)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Exemplos de Uso](#exemplos-de-uso)

---

## Visão Geral

Este projeto implementa um sistema end-to-end de previsão de preços de ações utilizando **Long Short-Term Memory (LSTM)**, um tipo de rede neural recorrente especialmente eficaz para dados temporais.

| Item              | Valor                          |
|-------------------|-------------------------------|
| **Empresa**       | Apple Inc.                    |
| **Ticker**        | AAPL                          |
| **Período**       | 2018-01-01 → 2024-07-20       |
| **Janela (seq)**  | 60 dias úteis                 |
| **Arquitetura**   | LSTM × 3 camadas + Dense      |
| **Framework**     | TensorFlow / Keras 2.16       |
| **API**           | FastAPI + Uvicorn             |
| **Deploy**        | Docker + Docker Compose       |

---

## Arquitetura do Modelo LSTM

```
Input (60, 1)
    │
    ▼
LSTM(128, return_sequences=True)
Dropout(0.2) → BatchNorm
    │
    ▼
LSTM(64, return_sequences=True)
Dropout(0.2) → BatchNorm
    │
    ▼
LSTM(32)
Dropout(0.2)
    │
    ▼
Dense(16, relu)
    │
    ▼
Dense(1)  ← preço previsto (normalizado)
```

**Parâmetros:** ~120k | **Otimizador:** Adam (lr=1e-3) | **Loss:** MSE

---

## Estrutura do Projeto

```
lstm-stock-predictor/
├── app/
│   └── main.py              # FastAPI — endpoints REST
├── model/
│   ├── train.py             # Script de treinamento LSTM
│   ├── lstm_model.h5        # Modelo treinado (gerado)
│   ├── scaler.pkl           # MinMaxScaler (gerado)
│   ├── model_meta.json      # Metadados (gerado)
│   └── metrics.json         # Métricas de avaliação (gerado)
├── notebooks/
│   └── EDA_e_Treinamento.ipynb
├── monitoring/
│   └── prometheus.yml
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

## Configuração

### Pré-requisitos

- Python 3.11+
- Docker & Docker Compose (para deploy)

### Instalação local

```bash
git clone https://github.com/seu-usuario/lstm-stock-predictor.git
cd lstm-stock-predictor

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Pipeline de Treinamento

```bash
# Treina o modelo e salva os artefatos em model/
make train

# Ou diretamente:
python model/train.py
```

### Etapas do pipeline

| Etapa | Descrição |
|-------|-----------|
| **1. Coleta** | Download automático via `yfinance` |
| **2. Pré-processamento** | Normalização MinMax, criação de janelas deslizantes (seq=60) |
| **3. Split** | Train 80% / Val 10% / Test 10% |
| **4. Modelo** | LSTM empilhado com Dropout e BatchNorm |
| **5. Treinamento** | EarlyStopping + ReduceLROnPlateau |
| **6. Avaliação** | MAE, RMSE, MAPE, R² no conjunto de teste |
| **7. Persistência** | `.h5` (modelo) + `.pkl` (scaler) + `.json` (meta/métricas) |

---

## API REST

### Iniciar localmente

```bash
make run
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Endpoints

| Método | Path | Descrição |
|--------|------|-----------|
| `GET`  | `/` | Status da API |
| `GET`  | `/health` | Health check (uptime, erros, latência) |
| `GET`  | `/metrics` | Métricas de treinamento + monitoramento |
| `GET`  | `/model/info` | Info do modelo (shape, params, metadata) |
| `POST` | `/predict` | Previsão com preços fornecidos manualmente |
| `POST` | `/predict/auto` | Busca automática de preços via Yahoo Finance |

---

## Deploy com Docker

### Build e start

```bash
make docker-build
make docker-up

# Ou diretamente:
docker compose up -d
```

### Serviços

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `lstm-api` | 8000 | FastAPI + Uvicorn |
| `prometheus` | 9090 | Coleta de métricas |
| `grafana` | 3000 | Dashboards (admin / lstm1234) |

```bash
# Logs em tempo real
make docker-logs

# Parar tudo
make docker-down
```

---

## Monitoramento

A API registra automaticamente em `model/monitor.jsonl`:

- Timestamp de cada requisição
- Path e método HTTP
- Status code
- Latência em ms

O endpoint `/metrics` expõe:

```json
{
  "monitoring": {
    "total_requests": 42,
    "total_errors": 1,
    "error_rate": 2.38,
    "avg_latency_ms": 87.4,
    "p95_latency_ms": 145.2,
    "uptime_seconds": 3600
  }
}
```

**Prometheus + Grafana** estão configurados no `docker-compose.yml` para monitoramento visual.

---

## Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | `mean(|y - ŷ|)` | Erro médio em USD |
| **RMSE** | `sqrt(mean((y-ŷ)²))` | Penaliza outliers |
| **MAPE** | `mean(|y-ŷ|/y) × 100` | Erro relativo em % |
| **R²** | `1 - SS_res/SS_tot` | Variância explicada (0→1) |

---

## Exemplos de Uso

### POST /predict (preços manuais)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [182.5, 183.2, 181.7, 184.0, 185.1, 184.8,
               183.9, 186.2, 187.0, 186.5, 185.8, 187.3,
               188.1, 187.6, 189.0, 188.5, 190.2, 191.0,
               190.5, 192.3, 191.8, 193.0, 192.5, 194.1,
               193.7, 195.2, 194.8, 196.0, 195.5, 197.1,
               196.6, 198.2, 197.8, 199.0, 198.5, 200.1,
               199.7, 201.2, 200.8, 202.0, 201.5, 203.1,
               202.7, 204.2, 203.8, 205.0, 204.5, 206.1,
               205.6, 207.2, 206.8, 208.0, 207.5, 209.1,
               208.7, 210.2, 209.8, 211.0, 210.5, 212.1]
  }'
```

**Resposta:**
```json
{
  "symbol": "AAPL",
  "predictions": [213.47],
  "n_predictions": 1,
  "model_meta": {"symbol": "AAPL", "seq_len": 60, "trained_at": "..."},
  "latency_ms": 85.3,
  "timestamp": "2024-07-21T10:00:00"
}
```

### POST /predict/auto (busca automática)

```bash
curl -X POST http://localhost:8000/predict/auto \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 5}'
```

---

## Testes

```bash
make test
# ou
pytest tests/ -v
```

---

## Licença

MIT License — uso livre para fins acadêmicos e comerciais.
