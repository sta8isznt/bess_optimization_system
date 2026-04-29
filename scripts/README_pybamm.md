# BESS Market-Realistic Pipeline

## 1. Project purpose

This repository contains a Battery Energy Storage System (BESS) simulation pipeline for a default **2 MWh / 1 MW lithium-ion battery** operating on 15-minute market intervals.

The script does **not** directly solve the full electrochemical equations of a lithium-ion cell. Instead, it provides a fast **market and financial simulation layer** with simplified battery-physics constraints:

- power limit,
- energy capacity limit,
- state-of-charge bounds,
- C-rate limit,
- depth-of-discharge-per-step limit,
- temperature proxy,
- degradation-cost lookup table,
- state-of-health loss,
- infeasibility penalties,
- net-profit calculation.

The intended architecture is:

```text
Lab data / Oxford degradation data
        ↓
PyBaMM electrochemical simulation and calibration
        ↓
validated degradation surface
        ↓
degradation_lut.csv
        ↓
bess_full_market_realistic_pipeline.py
        ↓
dispatch, SoC, SoH, degradation cost, net profit, report
```

PyBaMM is intended to run offline. Its outputs are converted into a degradation lookup table that can be evaluated quickly inside the market simulator.


## Scope

The repository currently implements a dispatch-level simulation and financial evaluation layer. It does not solve electrochemical partial differential equations during market simulation. High-fidelity electrochemical modelling should be performed offline using PyBaMM, with the resulting degradation surface exported as a lookup table consumed by this script.

---

## 2. Main file

```text
bess_full_market_realistic_pipeline.py
```

The script performs the following sequence:

1. Loads market input: timestamp, price, and ambient/cell temperature.
2. Loads an existing degradation LUT if available.
3. Rebuilds a market-realistic LUT with a C-rate validity mask.
4. Generates a constraint-aware charge/discharge dispatch.
5. Evaluates SoC, SoH, degradation cost, infeasibility penalties, and profit.
6. Exports CSV files and, if Plotly is installed, an HTML visual report.

---

## 3. Main assumptions

The default battery is:

```text
E_nom_MWh = 2.0
P_max_MW  = 1.0
dt_h      = 0.25
```

Therefore:

```text
physical C-rate = P_max / E_nom = 1 MW / 2 MWh = 0.5C
max DoD per 15-minute step = 0.5 × 0.25 = 0.125 = 12.5%
```

This is why the script treats DoD values above `0.125` as invalid for one 15-minute dispatch step in the default 2 MWh / 1 MW configuration.

---

## 4. Core software architecture

```text
BESSConfig
    ↓
load_market_input()
    ↓
build_market_realistic_lut()
    ↓
make_constraint_aware_dispatch()
    ↓
run_bess_financial_layer()
    ↓
summarize_financial_result()
    ↓
CSV outputs + HTML report
```

### 4.1 `BESSConfig`

`BESSConfig` stores the asset parameters and engineering assumptions:

| Parameter group | Examples |
|---|---|
| Battery size | `E_nom_MWh`, `P_max_MW`, `dt_h` |
| Operating window | `soc_min`, `soc_max`, `soh_eol` |
| Efficiency | `eta_charge`, `eta_discharge` |
| Thermal proxy | `T_cell_min_C`, `T_cell_max_C`, `thermal_power_coeff_C_per_Crate` |
| Economics | `replacement_cost_eur_per_MWh_capacity` |
| Dispatch logic | price quantiles and power fraction |
| Fallback degradation | DoD exponent, temperature coefficient, C-rate exponent |
| Penalties | SoC, power, C-rate, temperature, simultaneous-operation, SoH penalties |

Derived properties include:

```python
replacement_total_cost_eur = replacement_cost_eur_per_MWh_capacity * E_nom_MWh
roundtrip_efficiency       = eta_charge * eta_discharge
physical_c_rate_from_power = P_max_MW / E_nom_MWh
effective_c_rate_max       = min(c_rate_max, physical_c_rate_from_power)
p_c_rate_limit_MW          = effective_c_rate_max * E_nom_MWh
max_dod_per_step           = effective_c_rate_max * dt_h
```

---

## 5. Market input layer

The function:

```python
load_market_input(input_dir)
```

searches for one of the following input files:

```text
market_input.csv
bess_market_input.csv
bess_market_input_realistic.csv
bess_dispatch_fixed.csv
bess_financial_result_fixed.csv
bess_dispatch.csv
bess_financial_result.csv
```

Required columns:

```text
timestamp
price_eur_per_MWh
```

Optional temperature columns:

```text
T_amb_C
T_cell_C
T_cell_C_used
```

If no temperature column exists, the script assumes:

```text
T_amb_C = 25.0 °C
```

If no market input exists, the script generates a synthetic 96-step one-day price profile.

---

## 6. Physics layer implemented directly in the script

The script uses simplified engineering formulas. These are not a substitute for PyBaMM, but they provide a fast dispatch-level approximation.

### 6.1 C-rate

The C-rate measures the battery power relative to energy capacity:

```text
C_rate = P / E_nom
```

In the script:

```python
c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
```

For the default system:

```text
1 MW / 2 MWh = 0.5C
```

This means the battery is a 2-hour system at maximum power.

---

### 6.2 Maximum DoD per market step

For a timestep of length `dt_h`:

```text
DoD_step = P × dt_h / E_nom
```

In the script:

```python
DoD_step = max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh
```

For the default battery:

```text
DoD_step_max = 0.5C × 0.25 h = 0.125 = 12.5%
```

This is the main physical consistency constraint used by the dispatch layer. A 2 MWh / 1 MW battery cannot move 25% of its energy capacity in one 15-minute interval without exceeding its 0.5C physical power limit.

---

### 6.3 State of charge update

The script updates SoC using a capacity-normalized energy balance:

```text
SoC_next =
    SoC
    + η_charge × P_charge × Δt / usable_capacity
    - P_discharge × Δt / (η_discharge × usable_capacity)
```

where:

```text
usable_capacity = E_nom × max(SoH, SoH_EOL)
```

In code:

```python
usable_capacity_MWh = cfg.E_nom_MWh * max(soh, cfg.soh_eol)

soc_next = (
    soc
    + cfg.eta_charge * P_ch_MW * cfg.dt_h / usable_capacity_MWh
    - P_dis_MW * cfg.dt_h / (cfg.eta_discharge * usable_capacity_MWh)
)
```

Physical meaning:

- Charging increases SoC, but not all grid energy reaches the cell because `η_charge < 1`.
- Discharging decreases SoC more than the exported AC energy because discharge losses require more internal battery energy.
- As SoH decreases, usable capacity decreases, so the same power movement creates a larger SoC swing.

---

### 6.4 Temperature proxy

The script uses a simplified thermal stress proxy:

```text
T_cell = T_amb + k_thermal × C_rate
```

In code:

```python
T_cell_C = T_amb_C + cfg.thermal_power_coeff_C_per_Crate * c_rate
```

This is not a full thermal PDE model. It is a dispatch-level approximation that penalizes high-power operation by increasing estimated cell temperature.

A PyBaMM-based thermal model can replace this proxy later.

---

### 6.5 Degradation-cost surface

The script builds a degradation surface over:

```text
DoD
temperature
C-rate
```

When an external LUT is unavailable, it uses a fallback semi-empirical surface:

```text
cost(dod, T, C_rate)
=
base_cost
× (dod / ref_dod)^dod_exponent
× exp(temp_coeff × (T - ref_temp))
× (C_rate / ref_c_rate)^c_rate_exponent
× high_DoD_stress
× high_temperature_stress
```

In code:

```python
dod_factor = (dod / cfg.ref_dod) ** cfg.dod_exponent
temp_factor = exp(cfg.temp_coeff_per_C * (temp_c - cfg.ref_temp_C))
c_rate_factor = (c_rate / cfg.ref_c_rate) ** cfg.c_rate_exponent
```

Physical interpretation:

| Term | Interpretation |
|---|---|
| DoD factor | Deeper cycling usually increases cycle ageing |
| Temperature factor | Higher temperature accelerates parasitic reactions such as SEI growth |
| C-rate factor | Higher current increases polarization, gradients, and mechanical/electrochemical stress |
| High-DoD stress | Diagnostic penalty above feasible market boundary |
| High-temperature stress | Additional stress at elevated temperature |

The output unit is:

```text
EUR per MWh throughput
```

---

### 6.6 SoH loss from degradation cost

The script converts degradation cost into SoH loss:

```text
SoH_drop =
    degradation_cost_eur × (1 - SoH_EOL) / replacement_total_cost_eur
```

In code:

```python
soh_drop = cost_eur * (1.0 - cfg.soh_eol) / cfg.replacement_total_cost_eur
```

Interpretation:

- `replacement_total_cost_eur` is the assumed cost of replacing the full BESS capacity.
- The economic end-of-life threshold is `SoH_EOL = 0.80`.
- The usable lifetime window is therefore `1.00 - 0.80 = 0.20`.
- Degradation cost is treated as consuming part of this lifetime budget.

---

### 6.7 Net profit

The financial layer calculates:

```text
gross_margin =
    discharge_revenue - charge_cost
```

where:

```text
charge_cost       = P_ch × Δt × price
discharge_revenue = P_dis × Δt × price
```

Then:

```text
net_profit =
    gross_margin - degradation_cost - infeasibility_penalty
```

In code:

```python
net_profit = gross_margin - degradation_cost - penalty
```

This is the economic core of the pipeline.

---

## 7. Constraint-aware dispatch layer

The function:

```python
make_constraint_aware_dispatch()
```

uses a simple price-quantile heuristic:

```text
if price <= low_price_quantile:
    charge
elif price >= high_price_quantile:
    discharge
else:
    idle
```

Default thresholds:

```text
low_price_quantile  = 0.25
high_price_quantile = 0.75
```

The dispatch is clipped by:

- maximum power,
- C-rate limit,
- SoC headroom,
- SoC floor,
- no simultaneous charge/discharge.

This is a **heuristic simulator**, not yet a full mathematical optimization engine. A later version can replace this with a MILP or convex optimization layer.

---

## 8. Degradation LUT logic

The degradation LUT keeps two ideas separate:

### 8.1 Full diagnostic surface

The script keeps a full diagnostic DoD grid:

```text
0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25
```

This is useful for showing degradation behavior beyond normal dispatch constraints.

### 8.2 Optimizer-valid surface

For the default 2 MWh / 1 MW battery, only points satisfying:

```text
C_rate = DoD / dt_h <= effective_c_rate_max
```

are valid.

Since:

```text
effective_c_rate_max = 0.5
dt_h = 0.25
```

the valid boundary is:

```text
DoD <= 0.125
```

Points above this are retained for diagnostics but marked invalid for dispatch.

---

## 9. Physical violations and penalties

The script explicitly checks infeasibility conditions:

| Violation | Meaning |
|---|---|
| `negative_power` | Charge or discharge power below zero |
| `simultaneous_charge_discharge` | Battery charges and discharges at the same time |
| `charge_power_exceeded` | Charge power exceeds inverter or C-rate limit |
| `discharge_power_exceeded` | Discharge power exceeds inverter or C-rate limit |
| `C_rate_exceeded` | Power-to-capacity ratio exceeds allowed C-rate |
| `DoD_step_exceeded` | One-step energy movement exceeds physical limit |
| `SoC_current_out_of_bounds` | Current SoC outside allowed operating window |
| `SoC_next_out_of_bounds` | Next SoC outside allowed operating window |
| `cell_temperature_too_low` | Estimated cell temperature too low |
| `cell_temperature_too_high` | Estimated cell temperature too high |
| `SoH_increased` | Non-physical SoH increase |
| `EOL_reached` | SoH fell below end-of-life threshold |

The penalty layer makes infeasible dispatch economically unattractive.

---

## 10. PyBaMM physics layer: where it fits

PyBaMM should be used upstream of this script.

### 10.1 What PyBaMM provides

PyBaMM is an open-source Python framework for physics-based battery modelling. It provides:

- a framework for solving battery differential equations,
- a library of battery models,
- parameter sets,
- experiment definitions,
- solvers and visualization tools.

Relevant model families include:

| Model | Meaning | Use case |
|---|---|---|
| SPM | Single Particle Model | Fast reduced-order simulation |
| SPMe | Single Particle Model with Electrolyte | Compromise between speed and fidelity |
| DFN / P2D | Doyle-Fuller-Newman model | High-fidelity porous-electrode simulation |

For this project, the recommended offline physics layer is:

```python
model = pybamm.lithium_ion.DFN({
    "SEI": "solvent-diffusion limited",
    "lithium plating": "partially reversible",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
})
parameter_values = pybamm.ParameterValues("OKane2022")
```

This is more detailed than the market script and can be used to generate a degradation surface.

---

## 11. Electrochemical physics behind PyBaMM

A PyBaMM DFN/P2D model is based on porous-electrode theory and includes the major internal battery processes that the market script approximates externally.

### 11.1 Lithium diffusion in solid particles

Lithium concentration in electrode particles evolves through diffusion. In spherical particles, this is commonly represented by Fickian diffusion:

```text
∂c_s/∂t = (1/r²) ∂/∂r (D_s r² ∂c_s/∂r)
```

Physical meaning:

- lithium ions insert into and leave active material particles,
- concentration gradients form during high-power operation,
- stronger gradients create mechanical stress and degradation risk.

This is the physics behind why higher C-rate operation is more damaging.

---

### 11.2 Electrolyte transport

The electrolyte carries Li+ ions between electrodes. The DFN model tracks:

- electrolyte concentration,
- electrolyte potential,
- ionic current,
- transport limitations through porous media.

Physical meaning:

- high current causes electrolyte concentration gradients,
- electrolyte depletion and polarization can occur,
- local overpotentials can encourage lithium plating.

---

### 11.3 Charge conservation

The DFN model enforces current conservation in both phases:

```text
solid current + electrolyte current = applied current
```

Physical meaning:

- electrons move through the solid/electrode/current collector pathway,
- ions move through the electrolyte,
- electrochemical reactions transfer charge between these paths at the particle-electrolyte interface.

---

### 11.4 Butler-Volmer reaction kinetics

At the interface between active material and electrolyte, lithium intercalation is governed by electrochemical reaction kinetics. The common kinetic law is Butler-Volmer:

```text
j = j0 [exp(α_a F η / RT) - exp(-α_c F η / RT)]
```

Physical meaning:

- reaction current depends on overpotential,
- high currents require larger overpotentials,
- large overpotentials increase degradation risk,
- at low temperature or high charging current, lithium plating becomes more likely.

---

### 11.5 SEI growth

The solid-electrolyte interphase (SEI) forms on the negative electrode. It is partly protective but consumes cyclable lithium and electrolyte.

Main effects:

- loss of lithium inventory,
- capacity fade,
- impedance rise,
- increased heat generation and power fade.

SEI growth is one of the major mechanisms linking cycling and calendar ageing.

---

### 11.6 Lithium plating

Lithium plating occurs when metallic lithium deposits on the negative electrode instead of intercalating into graphite.

It is promoted by:

- fast charging,
- low temperature,
- high SoC,
- large overpotential,
- electrolyte transport limitations.

Effects:

- capacity loss,
- safety risk,
- nonlinear ageing,
- potential dendrite formation.

---

### 11.7 Particle cracking and loss of active material

High current and deep cycling generate concentration gradients inside particles. These gradients cause mechanical stress, which can fracture active material.

Consequences:

- fresh surfaces are exposed,
- more SEI forms,
- active material becomes electrically isolated,
- capacity and power fade increase.

---

## 12. Why the market script uses a LUT instead of running PyBaMM online

Running PyBaMM inside every market optimization step would be computationally expensive, especially for:

- long time horizons,
- scenario analysis,
- many assets,
- portfolio-level dispatch,
- stochastic price simulations.

Therefore, the correct architecture is:

```text
Offline:
    PyBaMM + lab data → degradation surface

Online / fast simulation:
    market dispatch → LUT lookup → degradation cost
```

This is the same modelling philosophy used in many engineering systems:

```text
high-fidelity simulator → calibrated surrogate / lookup table → fast optimizer
```

---

## 13. How to build the degradation LUT with PyBaMM

Recommended grid:

```text
DoD:        0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25
C-rate:     0.25C, 0.5C, 1.0C
Temperature: 10°C, 20°C, 25°C, 30°C, 35°C, 40°C
SoC window: optional extension
```

For each operating point:

1. Run a PyBaMM experiment with the target C-rate, DoD, and temperature.
2. Extract capacity fade or internal degradation variables.
3. Convert degradation to SoH loss.
4. Convert SoH loss to economic cost.
5. Export a row:

```text
dod, temperature_c, c_rate, deg_cost_eur_per_MWh_throughput, soh_drop
```

Suggested output:

```text
degradation_lut.csv
```

Example schema:

```text
dod
temperature_c
dt_h
c_rate
valid
deg_cost_eur_per_MWh_throughput
deg_cost_final
soh_drop
source
```

---

## 14. How Oxford/lab data should be used

Oxford battery degradation data should be used as the empirical validation layer.

Recommended validation targets:

| Target | Why it matters |
|---|---|
| Capacity fade vs cycles | Validates SoH trajectory |
| Voltage curve | Validates electrochemical behavior |
| Internal resistance / impedance trend | Validates power fade |
| Temperature sensitivity | Validates thermal-stress assumptions |
| OCV or diagnostic curves | Helps identify degradation modes such as LLI and LAM |

The purpose is not to perfectly reproduce every cell. The purpose is to show that your degradation-cost surface is anchored in real electrochemical behavior.

---

## 15. Full pipeline proposal

```text
1. Clean Oxford/lab data
        ↓
2. Fit/validate PyBaMM model
        ↓
3. Simulate operating grid:
       DoD × C-rate × temperature
        ↓
4. Extract:
       capacity loss, LLI, LAM, SEI loss, plating loss, throughput
        ↓
5. Convert physical degradation to:
       EUR/MWh throughput
        ↓
6. Export:
       degradation_lut.csv
        ↓
7. Run:
       bess_full_market_realistic_pipeline.py
        ↓
8. Output:
       dispatch, SoC, SoH, degradation cost, net profit
```

---

## 16. Running the market script

Basic command:

```bash
python bess_full_market_realistic_pipeline.py
```

With explicit input/output folders:

```bash
python bess_full_market_realistic_pipeline.py --input-dir . --output-dir .
```

With custom BESS size:

```bash
python bess_full_market_realistic_pipeline.py \
  --E-nom-MWh 2 \
  --P-max-MW 1 \
  --dt-h 0.25 \
  --c-rate-max 0.5
```

For a 4 MWh / 2 MW battery:

```bash
python bess_full_market_realistic_pipeline.py \
  --E-nom-MWh 4 \
  --P-max-MW 2 \
  --dt-h 0.25 \
  --c-rate-max 0.5
```

---

## 17. Expected outputs

The script writes:

| Output file | Description |
|---|---|
| `bess_market_input_market_realistic.csv` | Cleaned market input |
| `degradation_lut_market_realistic.csv` | Final degradation LUT with validity mask |
| `bess_dispatch_market_realistic.csv` | Dispatch schedule |
| `bess_financial_result_market_realistic.csv` | Full financial and physical result |
| `bess_financial_summary_market_realistic.csv` | One-row summary |
| `bess_telemetry_market_realistic.csv` | Compact telemetry table |
| `pipeline_market_realistic_manifest.json` | Run metadata and configuration |
| `bess_market_realistic_visual_report.html` | Interactive visual report, if Plotly is installed |

---

## 18. Current limitations

The current script is a simulation prototype with the following limitations:

1. **Dispatch is heuristic**, not a true optimization problem.
   - Current logic uses price quantiles.
   - Future version should use MILP or convex optimization.

2. **Thermal model is simplified**.
   - Current model uses `T_cell = T_amb + k × C_rate`.
   - Future version should use PyBaMM thermal outputs or a calibrated pack-level thermal model.

3. **Degradation surface is semi-empirical unless an external LUT exists**.
   - The fallback cost function is a placeholder when no calibrated LUT is available.
   - A production version should use PyBaMM + lab-calibrated degradation data.

4. **Battery pack aggregation is simplified**.
   - The script models a BESS as one aggregate asset.
   - Real systems include cells, modules, racks, PCS, HVAC, auxiliary load, and balancing logic.

5. **No degradation memory/path dependence yet**.
   - Real degradation depends on history, not only current DoD, C-rate, and temperature.
   - Future LUTs can include mean SoC, rest time, cycle sequence, and calendar ageing.

---

## 19. Recommended next improvements

### 19.1 Replace heuristic dispatch with optimization

Add a MILP layer:

```text
maximize:
    revenue - charge cost - degradation cost - penalty

subject to:
    SoC balance
    power limits
    C-rate limits
    no simultaneous charge/discharge
    valid degradation-LUT region
    terminal SoC constraint
```

### 19.2 Add PyBaMM LUT builder

Create:

```text
build_pybamm_degradation_lut.py
```

Purpose:

```text
Run PyBaMM experiments over DoD × C-rate × temperature grid
Extract degradation outputs
Convert to EUR/MWh throughput
Export degradation_lut.csv
```

### 19.3 Add Oxford validation notebook

Create:

```text
notebooks/01_validate_pybamm_against_oxford.ipynb
```

Plots:

```text
real capacity fade vs PyBaMM capacity fade
real voltage curve vs PyBaMM voltage curve
real SoH vs simulated SoH
degradation cost vs DoD
```

### 19.4 Add scenario analysis

Run different battery sizes:

```text
1 MW / 1 MWh
1 MW / 2 MWh
2 MW / 4 MWh
5 MW / 10 MWh
```

Compare:

```text
gross margin
degradation cost
net profit
final SoH
number of cycles
feasible dispatch share
```

---

## 20. Technical summary

This repository implements a dispatch-level BESS simulation layer that combines market prices, operating constraints, simplified physics proxies, degradation-cost lookup tables, and financial accounting. The script is designed to consume degradation surfaces generated offline from PyBaMM simulations and laboratory data.

The current implementation should be interpreted as a market-realistic simulation prototype, not as a full electrochemical model and not yet as a mathematical dispatch optimizer.

---

## 21. References

### PyBaMM documentation and software

1. PyBaMM GitHub repository  
   https://github.com/pybamm-team/PyBaMM

2. PyBaMM project website  
   https://pybamm.org/

3. PyBaMM DFN model documentation  
   https://docs.pybamm.org/en/stable/source/examples/notebooks/models/DFN.html

4. PyBaMM coupled degradation example  
   https://docs.pybamm.org/en/latest/source/examples/notebooks/models/coupled-degradation.html

5. PyBaMM experiment tutorial  
   https://docs.pybamm.org/en/latest/source/examples/notebooks/getting_started/tutorial-5-run-experiments.html

6. PyBaMM parameter sets documentation  
   https://docs.pybamm.org/en/stable/source/api/parameters/parameter_sets.html

### Useful papers and datasets

1. Sulzer, V. et al. (2021). *Python Battery Mathematical Modelling (PyBaMM).* Journal of Open Research Software.  
   https://doi.org/10.5334/jors.309

2. Howey, D. & Birkl, C. (2017). *Oxford Battery Degradation Dataset 1.* University of Oxford.  
   https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac

3. Birkl, C. R. et al. (2017). *Degradation diagnostics for lithium ion cells.* Journal of Power Sources.  
   https://doi.org/10.1016/j.jpowsour.2016.12.011

4. Edge, J. S. et al. (2021). *Lithium ion battery degradation: what you need to know.* Physical Chemistry Chemical Physics.  
   https://doi.org/10.1039/D1CP00359C

5. O'Kane, S. E. J. et al. (2022). *Lithium-ion battery degradation: how to model it.* Physical Chemistry Chemical Physics.  
   https://doi.org/10.1039/D2CP00417H

6. Marquis, S. G. et al. (2019). *An asymptotic derivation of a single particle model with electrolyte.* SIAM Journal on Applied Mathematics.  
   https://doi.org/10.1137/18M1189579

---

## 22. Interpretation

The current file is best described as:

```text
A dispatch-level BESS market simulation layer with simplified physics constraints,
designed to consume a PyBaMM/lab-derived degradation lookup table.
```

The electrochemical component should be implemented separately as:

```text
PyBaMM degradation LUT builder + Oxford validation notebook
```

The complete system then consists of:

```text
physics-calibrated degradation model
        +
market-aware BESS dispatch simulator
```
