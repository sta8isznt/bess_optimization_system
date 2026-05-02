# BESS Optimizer Dashboard

Dark Streamlit dashboard for the Greek Day-Ahead Market BESS optimizer.

The dashboard keeps the UI layer in `dashboard/`, while all optimizer,
forecasting, LUT-loading, scaling, and settlement logic is shared from
`src/bess_optimization/`.

## Run

From the repository root:

```bash
streamlit run dashboard/app.py
```

If your shell cannot find the `streamlit` console script after installation, use:

```bash
python3 -m streamlit run dashboard/app.py
```

## What It Shows

- Editable optimizer inputs for battery size, efficiencies, SoC limits, terminal SoC rule, price file, and degradation LUT.
- Headline KPIs for solver status, profit, revenue, purchase cost, degradation cost, energy moved, final SoC, and equivalent cycles.
- Three-panel dispatch view: DAM price with buy/sell windows, charge/discharge power bars, and SoC trajectory.
- Financial waterfall, constraint summary, clean dispatch table, and CSV downloads.
- Daily scenario comparison for degradation assumptions, one-hour/four-hour duration cases, and a no-degradation comparison case.
- A simple `Forecast Optimizer` page that builds the next-day DAM forecast, writes the optimizer-compatible forecast CSV, and solves the daily BESS dispatch on the forecast prices.

## Required Files

The default demo expects:

- `data/cleaned_data/price_signals_15m.csv`
- `data/cleaned_data/Reduced_LUT_Final.csv`
- `data/cleaned_data/Reduced_LUT_PyBaMM_Only.csv`

The app infers common timestamp and price column names, but the default price schema is:

```text
DELIVERY_MTU,DAM_Price_EUR_MWh
```

The default optimizer LUT schema is:

```text
energy,temperature_c,deg_cost_eur_per_MWh_throughput
```

## Demo Workflow

1. Select a date from `price_signals_15m.csv`.
2. Select `pybamm_only` and the PyBaMM-only LUT.
3. Run optimization.
4. Inspect the dispatch chart: charge at low prices, discharge at high prices, idle when spreads are too small.
5. Compare scenarios.
6. Download the schedule and summary KPIs.

## Forecast Workflow

1. Open `Forecast Optimizer` from the Streamlit page switcher.
2. Select the historical DAM price file and delivery date.
3. Click `Run`.
4. Inspect the basic KPIs, forecast-vs-real price chart, and dispatch table.
