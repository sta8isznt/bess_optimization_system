"""Build segment-level operating features for the Oxford degradation dataset.

The target is SoH_loss, defined as positive when measured capacity decreases.
No empirical degradation-cost coefficients are fitted here.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bess_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "bess_xdg_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROFILE_COLUMNS = [
    "time_s",
    "current_A",
    "voltage_V",
    "cell_temperature_C",
    "environment_temperature_C",
]


def _time_weighted_mean(values: pd.Series, delta_t_hours: pd.Series) -> float:
    valid = values.notna() & delta_t_hours.notna()
    denominator = delta_t_hours[valid].sum(skipna=True)
    if pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float((values[valid] * delta_t_hours[valid]).sum(skipna=True) / denominator)


def _load_clean_profile(profile_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(profile_path)
    raw_row_count = len(raw)
    profile = raw.dropna(how="all").copy()
    fully_empty_rows_dropped = raw_row_count - len(profile)

    for column in PROFILE_COLUMNS:
        profile[column] = pd.to_numeric(profile[column], errors="coerce")

    invalid_time_rows_dropped = int(profile["time_s"].isna().sum())
    profile = profile.dropna(subset=["time_s"]).copy()

    # Official plotData.m treats non-positive temperatures as no measurement.
    profile.loc[profile["cell_temperature_C"] <= 0, "cell_temperature_C"] = np.nan
    profile.loc[
        profile["environment_temperature_C"] <= 0, "environment_temperature_C"
    ] = np.nan

    metadata = {
        "profile_raw_rows": raw_row_count,
        "fully_empty_rows_dropped": fully_empty_rows_dropped,
        "invalid_time_rows_dropped": invalid_time_rows_dropped,
    }
    return profile, metadata


def build_segment_features(
    oxford_dir: Path, alignment_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build feature, summary, and QC dataframes."""
    alignment = pd.read_csv(alignment_path)
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    summary_inputs: list[dict[str, object]] = []

    for (strategy, cell), align_group in alignment.groupby(["strategy", "cell"], sort=True):
        cell = int(cell)
        stem = f"{strategy}_cell{cell}"
        capacity_path = oxford_dir / f"{stem}_capacityData.csv"
        profile_path = oxford_dir / f"{stem}_profileData.csv"
        if not capacity_path.exists():
            raise FileNotFoundError(f"Missing capacity file: {capacity_path}")
        if not profile_path.exists():
            raise FileNotFoundError(f"Missing profile file: {profile_path}")

        capacity = pd.read_csv(capacity_path)
        profile, profile_metadata = _load_clean_profile(profile_path)
        initial_capacity = float(capacity["capacity_Ah"].iloc[0])

        summary_inputs.append(
            {
                "strategy": strategy,
                "cell": cell,
                "initial_capacity_Ah": initial_capacity,
                "final_capacity_Ah": float(capacity["capacity_Ah"].iloc[-1]),
                **profile_metadata,
            }
        )

        for _, alignment_row in align_group.sort_values("interval").iterrows():
            interval = int(alignment_row["interval"])
            start_time = float(alignment_row["start_time_s_used"])
            end_time = float(alignment_row["end_time_s_used"])
            start_check = int(alignment_row["start_capacity_check"])
            end_check = int(alignment_row["end_capacity_check"])

            original_order = profile[
                (profile["time_s"] >= start_time) & (profile["time_s"] < end_time)
            ].copy()
            row_count = int(len(original_order))

            original_delta_t = original_order["time_s"].diff()
            duplicate_original_count = int((original_delta_t == 0).sum())
            backward_original_count = int((original_delta_t < 0).sum())

            segment = original_order.sort_values("time_s", kind="mergesort").reset_index(drop=True)
            forward_delta_s = segment["time_s"].shift(-1) - segment["time_s"]
            positive_delta_mask = forward_delta_s > 0
            delta_t_hours = forward_delta_s.where(positive_delta_mask) / 3600.0
            valid_delta_t_count = int(positive_delta_mask.sum())
            excluded_nonpositive_count = (
                int((forward_delta_s.iloc[:-1] <= 0).sum()) if row_count >= 2 else 0
            )

            current = segment["current_A"]
            abs_current = current.abs()
            voltage = segment["voltage_V"]
            cell_temperature = segment["cell_temperature_C"]

            sum_abs_current_dt = float((abs_current * delta_t_hours).sum(skipna=True))
            sum_current_squared_dt = float(((current**2) * delta_t_hours).sum(skipna=True))
            high_temp_exposure = float(
                (np.maximum(cell_temperature - 30.0, 0.0) * delta_t_hours).sum(skipna=True)
            )
            high_voltage_exposure = float(
                (np.maximum(voltage - 4.10, 0.0) * delta_t_hours).sum(skipna=True)
            )
            approximate_energy_throughput_wh = float(
                ((current * voltage).abs() * delta_t_hours).sum(skipna=True)
            )
            duration_hours = float(delta_t_hours.sum(skipna=True))

            capacity_start = float(capacity.loc[start_check, "capacity_Ah"])
            capacity_end = float(capacity.loc[end_check, "capacity_Ah"])
            capacity_loss = capacity_start - capacity_end
            soh_start = capacity_start / initial_capacity
            soh_end = capacity_end / initial_capacity
            soh_loss = soh_start - soh_end

            missing_cell_temp_rows = int(segment["cell_temperature_C"].isna().sum())
            missing_env_temp_rows = int(segment["environment_temperature_C"].isna().sum())
            missing_voltage_rows = int(segment["voltage_V"].isna().sum())
            missing_current_rows = int(segment["current_A"].isna().sum())
            missing_cell_temp_dt_hours = (
                float(delta_t_hours[segment["cell_temperature_C"].isna()].sum(skipna=True))
                if row_count
                else 0.0
            )
            missing_voltage_dt_hours = (
                float(delta_t_hours[segment["voltage_V"].isna()].sum(skipna=True))
                if row_count
                else 0.0
            )

            feature_rows.append(
                {
                    "strategy": strategy,
                    "cell": cell,
                    "interval": interval,
                    "start_capacity_check": start_check,
                    "end_capacity_check": end_check,
                    "start_profile_time_s": start_time,
                    "end_profile_time_s": end_time,
                    "capacity_start_Ah": capacity_start,
                    "capacity_end_Ah": capacity_end,
                    "capacity_loss_Ah": capacity_loss,
                    "SoH_start": soh_start,
                    "SoH_end": soh_end,
                    "SoH_loss": soh_loss,
                    "duration_hours": duration_hours,
                    "row_count": row_count,
                    "valid_delta_t_count": valid_delta_t_count,
                    "excluded_nonpositive_delta_t_count": excluded_nonpositive_count,
                    "sum_abs_current_dt": sum_abs_current_dt,
                    "sum_current_squared_dt": sum_current_squared_dt,
                    "high_temp_exposure": high_temp_exposure,
                    "high_voltage_exposure": high_voltage_exposure,
                    "avg_current_A": _time_weighted_mean(current, delta_t_hours),
                    "avg_abs_current_A": _time_weighted_mean(abs_current, delta_t_hours),
                    "avg_voltage_V": _time_weighted_mean(voltage, delta_t_hours),
                    "max_voltage_V": float(voltage.max(skipna=True)) if row_count else np.nan,
                    "avg_cell_temperature_C": _time_weighted_mean(
                        cell_temperature, delta_t_hours
                    ),
                    "max_cell_temperature_C": (
                        float(cell_temperature.max(skipna=True)) if row_count else np.nan
                    ),
                    "throughput_Ah": sum_abs_current_dt,
                    "approximate_energy_throughput_Wh": approximate_energy_throughput_wh,
                }
            )

            qc_flags: list[str] = []
            if duplicate_original_count:
                qc_flags.append("duplicate_timestamps")
            if backward_original_count:
                qc_flags.append("backward_timestamps_original_order")
            if excluded_nonpositive_count:
                qc_flags.append("nonpositive_delta_t_after_sort")
            if missing_cell_temp_rows:
                qc_flags.append("missing_cell_temperature")
            if missing_env_temp_rows:
                qc_flags.append("missing_environment_temperature")
            if missing_voltage_rows:
                qc_flags.append("missing_voltage")
            if missing_current_rows:
                qc_flags.append("missing_current")
            if row_count == 0:
                qc_flags.append("missing_profile_rows")

            qc_rows.append(
                {
                    "strategy": strategy,
                    "cell": cell,
                    "interval": interval,
                    "row_count": row_count,
                    "duplicate_timestamps_original_order": duplicate_original_count,
                    "backward_timestamps_original_order": backward_original_count,
                    "excluded_nonpositive_delta_t_count": excluded_nonpositive_count,
                    "missing_cell_temperature_rows": missing_cell_temp_rows,
                    "missing_environment_temperature_rows": missing_env_temp_rows,
                    "missing_voltage_rows": missing_voltage_rows,
                    "missing_current_rows": missing_current_rows,
                    "missing_cell_temperature_dt_hours": missing_cell_temp_dt_hours,
                    "missing_voltage_dt_hours": missing_voltage_dt_hours,
                    "flags": ";".join(qc_flags),
                }
            )

    features = (
        pd.DataFrame(feature_rows)
        .sort_values(["strategy", "cell", "interval"])
        .reset_index(drop=True)
    )
    qc = pd.DataFrame(qc_rows).sort_values(["strategy", "cell", "interval"]).reset_index(drop=True)
    summary_base = pd.DataFrame(summary_inputs).sort_values(["strategy", "cell"]).reset_index(drop=True)

    summary = features.groupby(["strategy", "cell"], as_index=False).agg(
        segment_count=("interval", "count"),
        total_duration_hours=("duration_hours", "sum"),
        total_row_count=("row_count", "sum"),
        total_valid_delta_t_count=("valid_delta_t_count", "sum"),
        total_excluded_nonpositive_delta_t_count=(
            "excluded_nonpositive_delta_t_count",
            "sum",
        ),
        total_throughput_Ah=("throughput_Ah", "sum"),
        total_energy_throughput_Wh=("approximate_energy_throughput_Wh", "sum"),
        total_high_voltage_exposure=("high_voltage_exposure", "sum"),
        total_high_temp_exposure=("high_temp_exposure", "sum"),
        total_capacity_loss_Ah=("capacity_loss_Ah", "sum"),
        total_SoH_loss=("SoH_loss", "sum"),
        mean_SoH_loss=("SoH_loss", "mean"),
    )
    summary = summary.merge(summary_base, on=["strategy", "cell"], how="left")
    summary["final_SoH"] = summary["final_capacity_Ah"] / summary["initial_capacity_Ah"]

    return features, summary, qc


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None.\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_notes(path: Path) -> None:
    """Write assumptions and limitations for the feature dataset."""
    path.write_text(
        "Oxford segment operating features notes\n"
        "=======================================\n\n"
        "Alignment rule: for interval j, use profileData rows where "
        "capacityData.profile_time_s[j] <= profileData.time_s < "
        "capacityData.profile_time_s[j+1].\n"
        "Fully empty profile rows are dropped. Rows with invalid/missing time_s "
        "cannot be assigned to segments and are dropped.\n"
        "cell_temperature_C <= 0 and environment_temperature_C <= 0 are treated "
        "as missing, following plotData.m.\n"
        "Within each segment, rows are sorted by time_s before calculating forward "
        "delta_t. Only positive forward delta_t contributes to integrated exposures.\n"
        "Duplicate timestamps produce zero delta_t after sorting and are excluded "
        "from integrated exposure calculations. Backward timestamps are reported "
        "from original file order.\n"
        "Averages are time-weighted using positive forward delta_t because "
        "profileData sampling is irregular.\n"
        "SoH_loss is the degradation target and is positive when capacity decreases.\n"
        "No alpha/beta/gamma/delta/epsilon coefficients are fitted in this step.\n"
        "No reliable SoC column exists in the source CSV files. high_SOC_exposure "
        "is therefore not computed here and should be handled later via "
        "coulomb-counting or a documented voltage proxy.\n"
    )


def write_quality_report(path: Path, features: pd.DataFrame, summary: pd.DataFrame, qc: pd.DataFrame) -> None:
    """Write a readable Markdown quality report without optional dependencies."""
    report: list[str] = []
    report.append("# Oxford Segment Operating Features Quality Report\n")
    report.append(
        "This report covers the generated segment-level operating feature dataset. "
        "No degradation coefficients were fitted.\n"
    )
    report.append("## Output Shape\n")
    report.append(f"- Segment feature rows: {len(features)}")
    report.append(f"- Strategy/cell summary rows: {len(summary)}")
    report.append(f"- QC rows: {len(qc)}")
    report.append(f"- Intervals with any QC flag: {int((qc['flags'].fillna('') != '').sum())}\n")

    report.append("## Strategy/Cell Summary\n")
    summary_columns = [
        "strategy",
        "cell",
        "segment_count",
        "total_duration_hours",
        "total_throughput_Ah",
        "total_SoH_loss",
        "final_SoH",
        "total_excluded_nonpositive_delta_t_count",
    ]
    report.append(_markdown_table(summary[summary_columns]))

    report.append("## Timestamp Issues By Strategy/Cell\n")
    timestamp_summary = qc.groupby(["strategy", "cell"], as_index=False).agg(
        intervals_with_duplicate_timestamps=(
            "duplicate_timestamps_original_order",
            lambda values: int((values > 0).sum()),
        ),
        intervals_with_backward_timestamps=(
            "backward_timestamps_original_order",
            lambda values: int((values > 0).sum()),
        ),
        total_excluded_nonpositive_delta_t=("excluded_nonpositive_delta_t_count", "sum"),
    )
    report.append(_markdown_table(timestamp_summary))

    report.append("## Missing Measurements By Strategy/Cell\n")
    missing_summary = qc.groupby(["strategy", "cell"], as_index=False).agg(
        intervals_with_missing_cell_temperature=(
            "missing_cell_temperature_rows",
            lambda values: int((values > 0).sum()),
        ),
        missing_cell_temperature_rows=("missing_cell_temperature_rows", "sum"),
        intervals_with_missing_environment_temperature=(
            "missing_environment_temperature_rows",
            lambda values: int((values > 0).sum()),
        ),
        missing_environment_temperature_rows=("missing_environment_temperature_rows", "sum"),
        intervals_with_missing_voltage=(
            "missing_voltage_rows",
            lambda values: int((values > 0).sum()),
        ),
        missing_voltage_rows=("missing_voltage_rows", "sum"),
        intervals_with_missing_current=(
            "missing_current_rows",
            lambda values: int((values > 0).sum()),
        ),
        missing_current_rows=("missing_current_rows", "sum"),
    )
    report.append(_markdown_table(missing_summary))

    report.append("## Intervals With Duplicate Or Backward Timestamps\n")
    timestamp_intervals = qc[
        (qc["duplicate_timestamps_original_order"] > 0)
        | (qc["backward_timestamps_original_order"] > 0)
    ][
        [
            "strategy",
            "cell",
            "interval",
            "duplicate_timestamps_original_order",
            "backward_timestamps_original_order",
            "excluded_nonpositive_delta_t_count",
        ]
    ]
    report.append(_markdown_table(timestamp_intervals))

    report.append("## Intervals With Missing Voltage\n")
    missing_voltage = qc[qc["missing_voltage_rows"] > 0][
        ["strategy", "cell", "interval", "missing_voltage_rows", "missing_voltage_dt_hours"]
    ]
    report.append("No intervals have missing voltage rows.\n" if missing_voltage.empty else _markdown_table(missing_voltage))

    report.append("## Intervals With Missing Cell Temperature\n")
    missing_temperature = qc[qc["missing_cell_temperature_rows"] > 0][
        [
            "strategy",
            "cell",
            "interval",
            "missing_cell_temperature_rows",
            "missing_cell_temperature_dt_hours",
        ]
    ]
    report.append(_markdown_table(missing_temperature))

    report.append("## SoC Note\n")
    report.append(
        "No reliable SoC column exists in the source CSV files, so high_SOC_exposure "
        "is not computed in this dataset. It should be handled later through "
        "coulomb-counting or a documented voltage proxy.\n"
    )

    path.write_text("\n".join(report))


def write_plots(output_dir: Path, features: pd.DataFrame) -> None:
    """Write diagnostic plots used during feature construction."""
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 5))
    features.boxplot(column="SoH_loss", by="strategy", ax=ax, grid=False)
    ax.set_title("SoH Loss by Strategy")
    fig.suptitle("")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("SoH loss per capacity interval")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_soh_loss_by_strategy.png", dpi=160)
    plt.close(fig)

    def scatter_plot(x: str, y: str, xlabel: str, ylabel: str, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for strategy in sorted(features["strategy"].unique()):
            subset = features[features["strategy"] == strategy]
            ax.scatter(subset[x], subset[y], label=strategy, alpha=0.75, s=35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Strategy")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    scatter_plot(
        "throughput_Ah",
        "SoH_loss",
        "Throughput (Ah)",
        "SoH loss",
        "Throughput vs SoH Loss",
        "plot_throughput_Ah_vs_SoH_loss.png",
    )
    scatter_plot(
        "high_voltage_exposure",
        "SoH_loss",
        "High-voltage exposure ((V - 4.10)+ hours)",
        "SoH loss",
        "High-Voltage Exposure vs SoH Loss",
        "plot_high_voltage_exposure_vs_SoH_loss.png",
    )
    scatter_plot(
        "high_temp_exposure",
        "SoH_loss",
        "High-temperature exposure ((C - 30)+ hours)",
        "SoH loss",
        "High-Temperature Exposure vs SoH Loss",
        "plot_high_temp_exposure_vs_SoH_loss.png",
    )


def write_outputs(
    output_dir: Path, features: pd.DataFrame, summary: pd.DataFrame, qc: pd.DataFrame
) -> None:
    """Write all CSV, note, report, and plot outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "oxford_segment_operating_features.csv", index=False)
    summary.to_csv(output_dir / "oxford_segment_operating_features_summary_by_cell.csv", index=False)
    qc.to_csv(output_dir / "oxford_segment_operating_features_quality_report.csv", index=False)
    write_notes(output_dir / "oxford_segment_operating_features_notes.txt")
    write_quality_report(
        output_dir / "oxford_segment_operating_features_quality_report.md",
        features,
        summary,
        qc,
    )
    write_plots(output_dir, features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Oxford segment-level operating features and QC outputs."
    )
    parser.add_argument("--oxford-dir", type=Path, default=Path("data/oxford"))
    parser.add_argument(
        "--alignment",
        type=Path,
        default=Path("data/produced_data/oxford_capacity_profile_alignment_segments.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/produced_data"))
    args = parser.parse_args()

    features, summary, qc = build_segment_features(args.oxford_dir, args.alignment)
    write_outputs(args.output_dir, features, summary, qc)
    print(f"Wrote {len(features)} segment rows to {args.output_dir}")
    print(f"Wrote {len(summary)} strategy/cell summary rows")
    print(f"Wrote {len(qc)} quality-check rows")


if __name__ == "__main__":
    main()
