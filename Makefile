.PHONY: install lint format dummy-backtest annual-backtest clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src scripts optimization

format:
	black src scripts optimization

dummy-backtest:
	python optimization/run_dummy_optimization_test.py

annual-backtest:
	python optimization/run_annual_2025_backtest.py

clean:
	rm -rf data/produced_data optimization/dummy_outputs optimization/annual_outputs
	rm -rf .pytest_cache .coverage htmlcov logs
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
