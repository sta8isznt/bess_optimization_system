# Optimization

This module contains the BESS arbitrage optimizer and reproducible backtest
runners.

## Inputs

- `data/cleaned_data/price_signals_15m.csv`: 15-minute DAM price signal used by the default runs.
- `data/cleaned_data/Reduced_LUT_Final.csv`: degradation-cost lookup table.
- `config.py`: battery power, capacity, efficiency, SoC, and timestep settings.

## Run

Edit the settings at the top of `run_engine.py`, especially
`RUN_MODE = "daily"` or `RUN_MODE = "annual"`, then run that file from the IDE
or run:

```bash
python optimization/run_engine.py
```

The script writes local outputs under `optimization/daily_outputs/` or
`optimization/annual_outputs/`.

Benchmark comparisons are controlled with:

- `RUN_BENCHMARKS = True`
- `BENCHMARK_MODELS = ("perfect", "naive")`

When benchmarks are enabled, the reports include the current degradation-aware
MILP, a perfect-foresight no-degradation MILP, and a naive EMA heuristic. The
benchmark rows intentionally use zero degradation cost and are written to
separate benchmark schedule CSVs plus a benchmark comparison CSV.

## Forecast-Driven Dispatch Backtest

Use this when you want the optimizer to plan on predicted prices, but evaluate
the resulting dispatch against actual realized DAM prices:

```bash
python scripts/run_forecast_strategy_backtest.py \
  --input-file optimization/data/cleaned_data/price_signals_15m.csv \
  --backtest-days 30 \
  --window-days 30 \
  --degradation-source pybamm_only
```

The wrapper leaves the MILP optimizer unchanged. For each target day it:

1. builds a next-day forecast using only timestamps before the target day,
2. optimizes the BESS schedule on the forecast price series,
3. preserves that dispatch and recalculates settlement cashflows on actual prices.

Outputs are written under `optimization/forecast_backtest_outputs/`:

- `*_schedule.csv`: interval-level dispatch with forecast and actual cashflows.
- `*_daily_stats.csv`: one row per target day.
- `*_trade_stats.csv`: active operation blocks paired into a ledger. Rows with
  `sequence = buy->sell` are strict completed trades; `sell->buy` and unpaired
  rows are retained for P&L traceability but excluded from the strict completed
  buy/sell trade ratio.
- `*_summary.csv` and `*_report.md`: aggregate profitability, forecast-error,
  strict completed buy/sell trade ratios, all-ledger ratios, and park scale-up
  statistics.

## PHASE 3: 

### Decision Variables:

The scheduling problem is mathematically formulated as a Mixed-Integer Linear Programming (MILP) model using the PuLP framework, solved via the CBC engine.

* Power Variables (p_buy, p_sell): Continuous variables constrained between [0, P_max], representing the charging and discharging power (MW) for each 15-minute Market Time Unit (MTU).

* State of Charge (soc): Continuous variable tracking the battery's stored energy (MWh), bounded by the operational limits (10% - 90%).

* Operational Mode (u): Binary variable acting as a switch (1 = Charging/Buying, 0 = Discharging/Selling or Idling).

* SOS2 Weights (lambdas): Continuous variables [0, 1] acting as interpolation weights for the piecewise linear approximation of the degradation curve.

### Constraints: 

* Simultaneous Operation Exclusion
> To prevent the optimizer from buying and selling energy in the same MTU, mutual exclusivity is enforced using the binary switch u[t]:
> * `P_buy[t] <= P_max * u[t]`
> * `P_sell[t] <= P_max * (1 - u[t])`

* Energy Balance (SoC Dynamics)
> * The State of Charge for the current MTU is strictly dependent on the previous interval's SoC. This continuity constraint integrates charging and discharging efficiencies to account for energy losses: `SoC[t] = SoC[t-1] + (P_buy[t] * eta_ch - P_sell[t] / eta_dis) * dt`

* Degradation Cost Linearization (SOS2)
> Battery wear increases non-linearly with the depth of discharge. To prevent the model from becoming a computationally expensive Mixed-Integer Non-Linear Programming (MINLP) problem, we employ Special Ordered Sets of Type 2 (SOS2).
> * The solver matches the decision variable `P_sell[t]` to predefined energy breakpoints (derived from the Phase 1 LUT).
> * Because the degradation cost curve is mathematically convex, the solver naturally selects two adjacent interpolation weights (lambdas) to calculate the exact dynamic_deg_cost in euros, maintaining strict MILP linearity and lightning-fast execution times.


### Objective Function 

`Maximize ∑ [ (Price[t] * (P_sell[t] - P_buy[t]) * dt) - dynamic_deg_cost[t] ]`

### Αssumptions:

* Cycle Continuity

>  To ensure operational readiness for subsequent days and to facilitate continuous multi-day backtesting, a Cycle Continuity Constraint is assumed. This dictates that the State of Charge (SoC) at the end of the 24-hour optimization horizon must equal its initial value (SoC_96​	=SoC_0). This prevents the model from unrealistic energy depletion ('dumping') for short-term profit and ensures the battery remains in a neutral state for the next day's market signals

* Modular Architecture & Linear Scale-Up

> * We designed the optimization engine using a Modular Scaling (Per-Unit) architecture. The heavy MILP mathematical problem is solved exclusively for a "Base Module" of 1 MW / 2 MWh (representing a standard physical BESS container).

> * For the baseline MVP, we assume the BESS acts as a "Price-Taker". This means the optimal dispatch schedule (when to buy/sell) for a single container is mathematically identical to the optimal schedule of the entire fleet.

* Market Cannibalization & Bidding Strategy Evolution

> * While the current MILP engine utilizes the "Price-Taker" assumption for computational speed, our strategic framework actively recognizes the **Market Cannibalization Effect**.

> * Injecting massive volumes of energy  into the Greek Day-Ahead Market (DAM) will inevitably alter the market equilibrium :

> > * Over time, as more BESS capacity enters the Greek grid, this self-cannibalization will compress the DAM price spreads.
