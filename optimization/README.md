# Optimization

This module contains the BESS arbitrage optimizer and reproducible backtest
runners.

## Inputs

- `data/cleaned_data/price_signals_15m.csv`: 15-minute DAM price signal used by the default runs.
- `data/cleaned_data/Reduced_LUT_Final.csv`: degradation-cost lookup table.
- `config.py`: battery power, capacity, efficiency, SoC, and timestep settings.

## Run

Edit the settings at the top of `run_optimization_backtest.py`, especially
`RUN_MODE = "daily"` or `RUN_MODE = "annual"`, then run that file from the IDE.

The scripts write local outputs under:

- `optimization/daily_outputs/`
- `optimization/annual_outputs/`

Those folders are generated artefacts and are not committed.
