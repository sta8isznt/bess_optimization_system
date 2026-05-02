.PHONY: install lint format test backtest clean

install:
	python3 -m pip install -e ".[dev]"

lint:
	ruff check src dashboard tests

format:
	black src dashboard tests

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"

backtest:
	PYTHONPATH=src python3 -m bess_optimization.cli.run_engine

clean:
	rm -rf data/produced_data outputs
	rm -rf .pytest_cache .coverage htmlcov logs
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
