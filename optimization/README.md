# OPTIMIZATION 

## PHASE 1:  Data for Backtesting our optimized BESS Strategy 

* We downloaded data from EnEx for 2024-2025:

> * Historical Scope: We aggregated and unified DAM data covering the entire year of 2024 and the first quarter of 2025. This extensive dataset ensures that the algorithm is rigorously tested against full seasonal cycles, capturing the dynamics of both high-demand summer peaks and varied winter loads.

> * Time Alignment: Hourly price signals were transformed into **15-minute** Market Time Units (MTU) using the Forward Fill technique. This step was crucial to align our inputs with the modern 15-minute settlement requirements of the Independent Power Transmission Operator.

## PHASE 2: Constructing Physical & Techincal Parameters for MILP Problem: 

* The optimization framework is highly highly configurable and heavily parameterized to reflect real-world physical constraints and European market regulations. The core technical specifications of the BESS are defined in the `config.py` file.

* Βreakdown:

| Parameter | Value | Description & Technical Justification |
| :--- | :--- | :--- |
| `p_max` | **1.0** MW | The maximum charging/discharging power (Maximum Power) of the base module. |
| `e_max` | **2.0** MWh | The nominal capacity (Nominal Capacity) of the module.|
| `eta_ch` | **0.92** | Efficiency during charging (Charging Efficiency).|
| `eta_dis` | **0.92** | Efficiency during discharging (Discharging Efficiency). Their combination implies a realistic Round-Trip Efficiency (RTE) of approximately 85%. |
| `soc_min` | **0.1** (10%) | The lower state of charge bound (Lower SoC Bound).. |
| `soc_max` | **0.9** (90%) | The upper state of charge bound (Upper SoC Bound). Enforcing this range (10-90%) is critical for LFP batteries, as preventing deep discharge (0%) and overcharging (100%) drastically reduces degradation costs and maximizes the asset's lifecycle. |
| `soc_init`| **0.5** (50%) | The initial energy level (Initial SoC) of the battery at the start of the optimization window. |
| `DT` | **0.25** hrs | The algorithm's time step is set at 15 minutes.

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

* Objective Function - `Maximize ∑ [ (Price[t] * (P_sell[t] - P_buy[t]) * dt) - dynamic_deg_cost[t] ]`
