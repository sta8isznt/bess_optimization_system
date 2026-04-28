# Architecture Overview

## System Components

### 1. Digital Twin (`src/bess_twin/digital_twin/`)

**Purpose**: Generate synthetic, physics-consistent battery time series data.

**Components**:
- `simulator.py`: ECM-based battery physics simulator
  - Input: Current profile
  - Output: SoC, Voltage, Power traces
  
- `timegan.py`: TimeGAN model for synthetic generation
  - Learns temporal patterns from synthetic data
  - Generates new realistic sequences
  
- `physics.py`: Physics-informed loss functions
  - SoC balance constraints
  - Voltage bounds enforcement
  - Smoothness regularization
  
- `train.py`: Training orchestration
  - Combines adversarial loss with physics constraints
  - Handles checkpointing and logging

**Key Features**:
- No real data required (synthetic-only training)
- Physics-constrained generation
- Deterministic simulator for reproducibility

---

### 2. Optimizer (`src/bess_twin/optimizer/`)

**Purpose**: Learn optimal control policies for battery management.

**Components**:
- `objective.py`: Cost/reward function
  - Minimize energy costs
  - Account for degradation
  - Peak power reduction
  
- `constraints.py`: Operational limits
  - Max charge/discharge rates
  - SoC bounds
  - Power ramp limits
  
- `train.py`: Policy training
  - Neural network-based policy
  - Gradient-based optimization
  - Constraint projection

**Key Features**:
- Learns from synthetic trajectories
- Satisfies hard constraints
- Extensible to real control strategies

---

### 3. Dashboard (`dashboards/`, `src/bess_twin/dashboard/`)

**Purpose**: Real-time visualization and monitoring.

**Features**:
- SoC timeline plots
- Voltage/power profiles
- Performance metrics
- Model comparison views
- Streamlit-based web interface

---

## Data Flow

```
[Simulator] 
    ↓
[Synthetic Trajectories] 
    ↓
[TimeGAN Training + Physics Loss]
    ↓
[Generated Synthetic Data]
    ↓
[Optimizer Training]
    ↓
[Control Policy]
    ↓
[Dashboard Visualization]
```

---

## Configuration System

All configs are YAML-based in `configs/`:

- `base.yaml`: Global settings (seed, device, paths)
- `digital_twin.yaml`: Simulator + TimeGAN + physics params
- `optimizer.yaml`: Objective + constraints + training
- `dashboard.yaml`: UI settings

Load with:
```python
from src.bess_twin.utils.config import load_digital_twin_config
config = load_digital_twin_config()
```

---

## Key Design Decisions

1. **Separation of Concerns**: Digital twin, optimizer, and dashboard are independent modules
2. **Physics-First**: Constraints embedded in training loss, not just postprocessing
3. **Config-Driven**: All hyperparameters in YAML, no hardcoding
4. **Testable**: Unit tests for each component
5. **Simple Baseline**: Before complex models, establish reproducible baseline

---

## Extension Points

To add features:
- New simulator model → `simulator.py`
- New objective → `objective.py`
- Custom metrics → Add to evaluation module
- Real data → Use same interface, swap simulator with dataloader
