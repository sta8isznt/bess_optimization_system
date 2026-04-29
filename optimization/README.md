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

## PHASE 3: Constraints:

### Assumptions:

* Cycle Continuity

>  To ensure operational readiness for subsequent days and to facilitate continuous multi-day backtesting, a Cycle Continuity Constraint is assumed. This dictates that the State of Charge (SoC) at the end of the 24-hour optimization horizon must equal its initial value (SoC_96​	=SoC_0). This prevents the model from unrealistic energy depletion ('dumping') for short-term profit and ensures the battery remains in a neutral state for the next day's market signals

