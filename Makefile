.PHONY: help install test lint format run-digital-twin run-optimizer run-dashboard clean

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make test                 - Run unit tests"
	@echo "  make lint                 - Run linting checks"
	@echo "  make format               - Format code with black"
	@echo "  make generate-data        - Generate synthetic data"
	@echo "  make train-digital-twin   - Train digital twin"
	@echo "  make train-optimizer      - Train optimizer"
	@echo "  make dashboard            - Launch dashboard"
	@echo "  make full-pipeline        - Run full pipeline"
	@echo "  make clean                - Clean cache and outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/

generate-data:
	python scripts/generate_data.py --num-trajectories 100

train-digital-twin:
	python scripts/train_digital_twin.py

train-optimizer:
	python scripts/train_optimizer.py

dashboard:
	python scripts/run_dashboard.py

full-pipeline: generate-data train-digital-twin train-optimizer
	@echo "Full pipeline complete!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf logs/*
