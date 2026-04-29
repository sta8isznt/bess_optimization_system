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

* Cycle Continuity

>  To ensure operational readiness for subsequent days and to facilitate continuous multi-day backtesting, a Cycle Continuity Constraint is assumed. This dictates that the State of Charge (SoC) at the end of the 24-hour optimization horizon must equal its initial value (SoC_96​	=SoC_0). This prevents the model from unrealistic energy depletion ('dumping') for short-term profit and ensures the battery remains in a neutral state for the next day's market signals

### Objective Function 

`Maximize ∑ [ (Price[t] * (P_sell[t] - P_buy[t]) * dt) - dynamic_deg_cost[t] ]`
