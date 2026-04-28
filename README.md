# BESS Digital Twin + Optimizer

A battery energy storage system digital twin with physics-constrained TimeGAN and operational optimizer.

## Structure

```
bess-digital-twin/
├── configs/            # Configuration files
├── data/               # Raw, synthetic, processed data
├── models/             # Saved model checkpoints
├── src/bess_twin/      # Core library
│   ├── digital_twin/   # TimeGAN + physics simulator
│   ├── optimizer/      # Optimization module
│   ├── dashboard/      # Visualization
│   └── utils/          # Utilities
├── scripts/            # Entry point scripts
├── dashboards/         # Streamlit app
├── tests/              # Unit tests
└── notebooks/          # Exploration notebooks
```

## Quick Start

1. Generate synthetic data
   ```bash
   python scripts/generate_data.py
   ```

2. Train digital twin
   ```bash
   python scripts/train_digital_twin.py
   ```

3. Train optimizer
   ```bash
   python scripts/train_optimizer.py
   ```

4. View dashboard
   ```bash
   streamlit run dashboards/streamlit_app.py
   ```

## Components

### Digital Twin
- Physics-based simulator (battery dynamics)
- TimeGAN for synthetic data generation
- Physics loss constraints (SoC, voltage)

### Optimizer
- Cost/reward objective
- Operational constraints
- Optimization training

### Dashboard
- Real-time metrics
- Time series plots
- Model comparisons
