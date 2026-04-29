# Optimization

This module contains the BESS arbitrage optimizer and reproducible backtest
runners.

## Inputs

- `price_signals_15m.csv`: 15-minute DAM price signal used by the default runs.
- `Reduced_LUT_Final.csv`: degradation-cost lookup table.
- `config.py`: battery power, capacity, efficiency, SoC, and timestep settings.

Raw daily DAM Excel downloads under `Results_*` are treated as local inputs and
are ignored by git.

## Run

From the repository root:

```bash
python optimization/run_dummy_optimization_test.py
python optimization/run_annual_2025_backtest.py
```

The scripts write local outputs under:

- `optimization/dummy_outputs/`
- `optimization/annual_outputs/`

Those folders are generated artefacts and are not committed.
