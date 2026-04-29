"""Generate DoD/Temperature LUT using PyBaMM digital twin physical layer."""

import argparse
from pathlib import Path

import numpy as np

from src.bess_twin.digital_twin.physical_layer import (
    SimulationConfig,
    run_lut_grid,
)


def _parse_range(arg: str):
    """Parse a range string 'start,stop,step' into a numpy array."""
    parts = [float(p.strip()) for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError("Range must be 'start,stop,step'")
    start, stop, step = parts
    return np.arange(start, stop + 1e-9, step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dod-range",
        default="0.1,1.0,0.1",
        help="DoD grid as start,stop,step",
    )
    parser.add_argument(
        "--temp-range",
        default="10,45,5",
        help="Temperature grid (C) as start,stop,step",
    )
    parser.add_argument("--output", default="lut_dod_temp.csv", help="Output LUT filename")
    parser.add_argument(
        "--profile-minutes",
        type=float,
        default=15.0,
        help="Profile duration in minutes",
    )
    parser.add_argument("--capacity-ah", type=float, default=100.0)
    parser.add_argument("--voltage-nominal", type=float, default=48.0)
    parser.add_argument(
        "--replace-cost",
        type=float,
        default=60000.0,
        help="Replacement cost in EUR/MWh",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dod_values = _parse_range(args.dod_range)
    temp_values = _parse_range(args.temp_range)

    cfg = SimulationConfig(
        capacity_ah=args.capacity_ah,
        voltage_nominal=args.voltage_nominal,
        profile_minutes=args.profile_minutes,
        replace_cost_eur_per_mwh=args.replace_cost,
    )

    lut = run_lut_grid(dod_values, temp_values, cfg)
    out_path = args.output_dir / args.output
    lut.to_csv(out_path, index=False)
    print(f"Saved LUT to {out_path}")


if __name__ == "__main__":
    main()
