.PHONY: install train test run docker-build docker-up docker-down clean

## ─── Desenvolvimento ────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

train:
	cd model && python train.py

test:
	pytest tests/ -v --tb=short

## ─── API local ───────────────────────────────────────────────────────────────
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## ─── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker build -t lstm-stock-predictor:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f lstm-api

## ─── Limpeza ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f model/monitor.jsonl
