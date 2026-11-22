PY ?= .venv/bin/python
MONTHS ?= 2023-12 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONPATH := $(CURDIR)

.PHONY: install data ingest train serve grpc worker test lint proto backtest loadtest up down k8s

install:
	python3.13 -m venv .venv && $(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/fetch_month.py $(MONTHS)

ingest:
	$(PY) -m driftline.cli ingest 2024-01 2024-02 2024-03

train:
	$(PY) -m driftline.cli train --train 2024-01 2024-02 --eval 2024-03

serve:
	$(PY) -m driftline.cli serve --port 8000

grpc:
	$(PY) -m driftline.cli grpc

worker:
	DRIFTLINE_REDIS_URL=redis://localhost:6379/0 .venv/bin/celery -A driftline.monitoring.tasks worker --beat -l info

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

proto:
	$(PY) -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. driftline/serving/proto/predict.proto

backtest:
	$(PY) scripts/backtest.py --train-window 2
	$(PY) scripts/backtest.py --train-window 4

loadtest:
	$(PY) scripts/loadtest.py --month 2024-04 --requests 5000 --concurrency 32

up:
	docker compose up -d --build

down:
	docker compose down

k8s:
	kubectl apply -k deploy/k8s
