"""CLI entry point for forecast-driven dispatch backtesting."""

from __future__ import annotations

from bess_optimization.services.forecast_strategy_backtest import main


if __name__ == "__main__":
    raise SystemExit(main())
