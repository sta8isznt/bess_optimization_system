# 🧠 Degradation Cost Estimation Methodology

## 🔹 Core Principle

> Our framework requires a **parameterized battery model**, which can be obtained through:
> - manufacturer data (when available),
> - literature parameter sets (e.g. Chen2020),
>
> Once this model is available, we can systematically compute degradation cost under arbitrary operating conditions.

---

# ⚙️ 1. System Inputs & Economic Mapping

### Battery-level inputs:
- **C-rate\_max** = P / E  
- **SoH\_EoL** = End-of-life threshold (e.g. 80%)

### Economic calibration:
We define a direct mapping between degradation and cost:

0.01 SoH loss → 6,000 €

Thus:

degradation_cost_eur = deg_cost_per_MWh × throughput_MWh

👉 Linear cost mapping on throughput, while degradation itself is nonlinear.

---

# 🔄 2. Operating Regime Modeling

We simulate battery operation over **15-minute market intervals**:

- dt = 0.25 hours  

- throughput\_MWh = max(P_ch, P_dis) × dt  

- DoD\_step = energy_throughput / E_nom  

- C-rate = DoD / dt  

👉 To amplify degradation signal:
- Simulate **12 consecutive cycles (3 hours total)**

---

# 🌡️ 3. Scenario Space

We evaluate degradation across:

- DoD values  
- Temperature  

### Base assumption:
- Operating temperature = **25°C** (due to HVAC)

Temperature sweeps are used for:
- sensitivity analysis  
- LUT construction  

---

# 🔬 4. Physics-Based Battery Modeling

We use **PyBaMM (Python Battery Mathematical Modelling)**.

## 🔋 Model Choice

- **SPM (Single Particle Model)**  
  → Each electrode approximated as a single particle  
  → Captures lithium diffusion in solid phase  

- **SPMe (Single Particle Model with electrolyte)**  
  → Adds electrolyte dynamics  

- **DFN (Doyle-Fuller-Newman)**  
  → Full model (not used due to complexity)  

👉 We use **SPM/SPMe** for balance between accuracy and speed.

---

## 📦 Parameterization

We use the **Chen2020 parameter set**, which includes:

- Diffusion coefficients  
- Particle radii  
- Electrode thickness  
- Reaction rates  
- Conductivity  
- Initial concentrations  
- SEI degradation parameters  

⚠️ Note: This is a proxy model due to data scarcity.

---

# ⚡ 5. Physical Laws (Core of the Digital Twin)

The model solves **coupled nonlinear differential equations**:

---

## 🧪 (A) Lithium Diffusion (Fick’s Law)

∂c_s/∂t = (1/r²) ∂/∂r (D_s r² ∂c_s/∂r)

👉 Lithium transport inside particles

---

## 🔌 (B) Charge Conservation

∇·(σ ∇φ_s) = −j  

∇·(κ ∇φ_e) + diffusion terms = j  

👉 Current flow + electrochemical coupling

---

## ⚛️ (C) Butler–Volmer Kinetics

j = j₀ [exp(α_a F η / RT) − exp(−α_c F η / RT)]

👉 Reaction rate at electrode interface

---

## 🌡️ (D) Thermal Dynamics

ρ c_p dT/dt = Q_gen − hA(T − T_amb)

👉 Heat from:
- ohmic losses  
- reaction overpotentials  

---

## 🧱 (E) SEI Growth (Degradation)

dL_SEI/dt = f(j, T, c)

👉 Causes:
- lithium loss  
- resistance increase  
- capacity fade  

---

# 📉 6. From Physics → Degradation

We compute:

- SoH (capacity loss)  
- Internal resistance growth  

⚠️ Important:

SoH ≠ linear in DoD  

Depends on:
- DoD  
- C-rate  
- temperature  
- history  

---

# 💸 7. Degradation Cost Computation

We link physics to economics:

W_deg = (dE / dL) × C_EOL  

Where:
- L = capacity loss  
- E = energy throughput  
- C_EOL = replacement cost (€/kWh)  

Final:

degradation_cost = f(DoD, T)

---

# 📊 8. LUT Construction

We build a Look-Up Table:

- Rows: DoD  
- Columns: Temperature  
- Values: €/MWh  

### Process:

For each DoD:
- compute C-rate  
- simulate 12 cycles  
- measure SoH loss  
- convert to € cost  

---

# 🚀 9. Key Takeaways

- Physics-based (not heuristic)  
- Parameter-driven (portable)  
- Captures nonlinear degradation  
- Converts electrochemistry → € cost  
- Optimization-ready  

---

# 🧠 TL;DR

> We use a physics-informed battery digital twin to translate operating conditions (DoD, C-rate, temperature) into degradation cost in €/MWh, enabling economically optimal battery operation.