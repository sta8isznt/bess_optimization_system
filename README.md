# Battery Optimization in the Greek Electricity Market

Python toolkit for battery energy storage optimization, degradation-cost modeling,
and simple backtesting on 15-minute day-ahead market prices.

## Repository Layout

- `src/`: battery digital-twin, data-processing, and utility code.
- `optimization/`: MILP optimizer, backtest runners, price signals, and LUT input.
- `scripts/`: command-line entry points for supporting data/LUT generation.
- `data/oxford/`: Oxford battery dataset input files used by the degradation pipeline.

Generated outputs are intentionally not tracked. Recreate them by running the
scripts, and keep the GitHub repository focused on source and stable input data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Common Commands

Edit `RUN_MODE` near the top of `optimization/run_optimization_backtest.py`
to choose `"daily"` or `"annual"`, then run that file from the IDE.

## Generated Outputs

These paths are local artefacts and are ignored by git:

- `data/produced_data/`
- `optimization/daily_outputs/`
- `optimization/annual_outputs/`
- `optimization/Results_*/`

The tracked optimizer inputs are:

- `optimization/data/cleaned_data/price_signals_15m.csv`
- `optimization/data/cleaned_data/Reduced_LUT_Final.csv`
