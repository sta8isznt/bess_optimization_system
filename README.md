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

## Greek DAM 15-Minute Price Forecasting

The forecasting module produces a next-day 15-minute forecast for Greek
Day-Ahead Market prices. It is designed as a lightweight baseline for the BESS
optimizer, using recent slot-of-day seasonality blended with the previous day's
same 15-minute slot. It is intentionally simple and hackathon-ready, not a
production-grade market forecaster.

The strict target is 15-minute Market Time Unit data. Greek SDAC delivery moved
to 15-minute MTUs for delivery day 2025-10-01 onward, so older hourly data
should not be mixed in unless you explicitly run with
`--allow-hourly-upsampling` for testing.

Example:

```bash
python3 scripts/run_dam_15min_forecast.py \
  --input-file optimization/data/cleaned_data/price_signals_15m.csv \
  --target-date 2026-05-01 \
  --output-file optimization/data/forecasts/dam_15min_forecast_next_day.csv \
  --window-days 30 \
  --model seasonal
```

If `--target-date` is omitted, the script forecasts the day after the latest
timestamp in the input file. If `--input-file` is omitted, it tries to discover a
reasonable local DAM price file from the existing project data folders.

Expected historical input columns are inferred where possible. The current
cleaned project file uses:

```text
DELIVERY_MTU,DAM_Price_EUR_MWh
```

The main forecast output contains 96 rows. Key columns include:

```text
timestamp
forecast_price_eur_mwh
slot_id
date
hour
minute
day_of_week
is_weekend
model_name
created_at_utc
forecast_reason
```

The script also writes an optimizer-compatible CSV beside it:

```text
optimization/data/forecasts/dam_15min_forecast_next_day_optimizer_input.csv
```

Run the lightweight synthetic self-check with:

```bash
python3 scripts/run_dam_15min_forecast.py --self-check
```

Optionally backtest the baseline on the last complete historical days:

```bash
python3 scripts/run_dam_15min_forecast.py --backtest-days 7
```

## Generated Outputs

These paths are local artefacts and are ignored by git:

- `data/produced_data/`
- `optimization/daily_outputs/`
- `optimization/annual_outputs/`
- `optimization/Results_*/`

The tracked optimizer inputs are:

- `optimization/data/cleaned_data/price_signals_15m.csv`
- `optimization/data/cleaned_data/Reduced_LUT_Final.csv`
