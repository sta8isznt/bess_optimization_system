.PHONY: install lint format backtest clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src scripts optimization

format:
	black src scripts optimization

backtest:
	python optimization/run_engine.py

clean:
	rm -rf data/produced_data optimization/daily_outputs optimization/annual_outputs
	rm -rf .pytest_cache .coverage htmlcov logs
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
