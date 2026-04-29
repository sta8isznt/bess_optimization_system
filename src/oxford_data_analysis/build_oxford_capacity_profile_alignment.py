"""Build capacity-check to measured-profile alignment segments for Oxford data.

This script does not fit any degradation model. It only determines which
measured profile rows belong to each capacity-measurement interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _split_stem(stem: str) -> tuple[str, int]:
    strategy, cell_part = stem.split("_")
    return strategy, int(cell_part.replace("cell", ""))


def build_alignment_table(oxford_dir: Path) -> pd.DataFrame:
    """Create the segment alignment table from Oxford capacity/profile files."""
    rows: list[dict[str, object]] = []

    capacity_paths = sorted(oxford_dir.glob("*_capacityData.csv"))
    if not capacity_paths:
        raise FileNotFoundError(f"No *_capacityData.csv files found in {oxford_dir}")

    for capacity_path in capacity_paths:
        stem = capacity_path.name.replace("_capacityData.csv", "")
        strategy, cell = _split_stem(stem)
        profile_path = oxford_dir / f"{stem}_profileData.csv"
        if not profile_path.exists():
            raise FileNotFoundError(f"Missing matching profile file: {profile_path}")

        capacity = pd.read_csv(capacity_path)
        profile = pd.read_csv(profile_path)
        profile["time_s"] = pd.to_numeric(profile["time_s"], errors="coerce")
        profile = profile.dropna(subset=["time_s"]).copy()

        initial_capacity = float(capacity["capacity_Ah"].iloc[0])
        for interval in range(len(capacity) - 1):
            start_time = float(capacity["profile_time_s"].iloc[interval])
            end_time = float(capacity["profile_time_s"].iloc[interval + 1])
            segment = profile[
                (profile["time_s"] >= start_time) & (profile["time_s"] < end_time)
            ].copy()

            row_count = int(len(segment))
            duration_covered = (
                float(segment["time_s"].max() - segment["time_s"].min())
                if row_count >= 2
                else 0.0
            )
            expected_duration = end_time - start_time

            deltas = segment["time_s"].diff()
            duplicate_count = int((deltas == 0).sum())
            backward_count = int((deltas < 0).sum())
            gaps_gt_900_count = int((deltas > 900).sum())

            flags: list[str] = []
            if row_count == 0:
                flags.append("missing_profile_rows")
            if expected_duration <= 0:
                flags.append("zero_or_negative_expected_duration")
            if expected_duration > 0 and duration_covered / expected_duration < 0.95:
                flags.append("poor_coverage_lt_0.95")
            if duplicate_count:
                flags.append(f"duplicate_ts={duplicate_count}")
            if backward_count:
                flags.append(f"backward_ts={backward_count}")
            if gaps_gt_900_count:
                flags.append(f"gap_gt_900s={gaps_gt_900_count}")

            start_capacity = float(capacity["capacity_Ah"].iloc[interval])
            end_capacity = float(capacity["capacity_Ah"].iloc[interval + 1])
            delta_capacity = end_capacity - start_capacity

            rows.append(
                {
                    "strategy": strategy,
                    "cell": cell,
                    "interval": interval,
                    "start_capacity_check": interval,
                    "end_capacity_check": interval + 1,
                    "start_time_s_used": start_time,
                    "end_time_s_used": end_time,
                    "number_of_profileData_rows_in_interval": row_count,
                    "duration_covered_by_profileData_s": duration_covered,
                    "expected_interval_duration_s": expected_duration,
                    "coverage_ratio": (
                        duration_covered / expected_duration
                        if expected_duration > 0
                        else np.nan
                    ),
                    "start_capacity_Ah": start_capacity,
                    "end_capacity_Ah": end_capacity,
                    "delta_capacity_Ah": delta_capacity,
                    "delta_SoH": delta_capacity / initial_capacity,
                    "duplicate_timestamps_in_interval": duplicate_count,
                    "backward_timestamps_in_interval": backward_count,
                    "gaps_gt_900s_in_interval": gaps_gt_900_count,
                    "flags": ";".join(flags),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "cell", "interval"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Oxford capacity/profile alignment segments."
    )
    parser.add_argument("--oxford-dir", type=Path, default=Path("data/oxford"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/produced_data/oxford_capacity_profile_alignment_segments.csv"),
    )
    args = parser.parse_args()

    alignment = build_alignment_table(args.oxford_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(args.output, index=False)
    print(f"Wrote {len(alignment)} rows to {args.output}")


if __name__ == "__main__":
    main()
