"""Build the minimal Oxford degradation modeling table.

This table is the starting point for fitting an interpretable empirical
degradation function. It intentionally keeps only segment identifiers,
degradation targets, and the currently available integrated operating
features. It does not estimate SoC and does not fit model coefficients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_COLUMNS = [
    "strategy",
    "cell",
    "interval",
    "capacity_start_Ah",
    "capacity_end_Ah",
    "capacity_loss_Ah",
    "SoH_start",
    "SoH_end",
    "SoH_loss",
    "throughput_Ah",
    "sum_current_squared_dt",
    "high_temp_exposure",
    "high_voltage_exposure",
]

QC_COLUMNS = [
    "strategy",
    "cell",
    "interval",
    "row_count",
    "excluded_nonpositive_delta_t_count",
    "missing_cell_temperature_rows",
    "missing_voltage_rows",
    "flags",
]


def build_modeling_table(features_path: Path, qc_path: Path | None = None) -> pd.DataFrame:
    """Return a compact degradation modeling table.

    The table keeps only the target variables and the current set of available
    explanatory variables:
    - throughput_Ah maps to the integrated |I| term.
    - sum_current_squared_dt maps to the integrated I^2 term.
    - high_temp_exposure maps to the high-temperature exposure term.
    - high_voltage_exposure maps to the high-voltage exposure term.

    high_SOC_exposure is intentionally absent because the Oxford CSV files do
    not contain a reliable SoC column.
    """
    features = pd.read_csv(features_path)
    missing = [column for column in MODEL_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    model_table = features[MODEL_COLUMNS].copy()

    if qc_path is not None:
        qc = pd.read_csv(qc_path)
        available_qc_columns = [column for column in QC_COLUMNS if column in qc.columns]
        model_table = model_table.merge(
            qc[available_qc_columns],
            on=["strategy", "cell", "interval"],
            how="left",
        )

    return model_table.sort_values(["strategy", "cell", "interval"]).reset_index(drop=True)


def write_notes(path: Path) -> None:
    """Write short notes for the compact modeling table."""
    path.write_text(
        "Oxford degradation modeling table notes\n"
        "=======================================\n\n"
        "This compact table is intended as the starting point for an empirical, "
        "interpretable degradation-function fit.\n\n"
        "Target variables:\n"
        "- SoH_loss: main target. Positive when capacity decreases.\n"
        "- capacity_loss_Ah: capacity_start_Ah - capacity_end_Ah.\n\n"
        "Current explanatory variables:\n"
        "- throughput_Ah: integrated absolute current, sum |I| * dt_hours.\n"
        "- sum_current_squared_dt: integrated squared current, sum I^2 * dt_hours.\n"
        "- high_temp_exposure: sum max(0, cell_temperature_C - 30) * dt_hours.\n"
        "- high_voltage_exposure: sum max(0, voltage_V - 4.10) * dt_hours.\n\n"
        "high_SOC_exposure is not included because the source CSV files do not "
        "contain a reliable SoC column. It should be added later only after a "
        "documented SoC reconstruction step, such as coulomb-counting or a "
        "carefully validated voltage proxy.\n\n"
        "No alpha, beta, gamma, delta, or epsilon coefficients are fitted by this script.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact Oxford degradation modeling table."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/produced_data/oxford_segment_operating_features.csv"),
    )
    parser.add_argument(
        "--qc",
        type=Path,
        default=Path("data/produced_data/oxford_segment_operating_features_quality_report.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_modeling_table.csv"),
    )
    parser.add_argument(
        "--notes-output",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_modeling_table_notes.txt"),
    )
    parser.add_argument(
        "--without-qc",
        action="store_true",
        help="Do not append compact QC columns to the modeling table.",
    )
    args = parser.parse_args()

    qc_path = None if args.without_qc else args.qc
    model_table = build_modeling_table(args.features, qc_path=qc_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_table.to_csv(args.output, index=False)
    write_notes(args.notes_output)

    print(f"Wrote {len(model_table)} rows to {args.output}")
    print(f"Wrote notes to {args.notes_output}")


if __name__ == "__main__":
    main()
