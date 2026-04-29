"""Convert empirical Oxford SoH-loss predictions into degradation cost.

The conversion is intentionally separated from model fitting. It maps predicted
SoH loss to EUR using an end-of-life replacement-cost convention:

    cost_eur = (delta_soh / eol_loss_fraction)
               * replacement_cost_eur_per_mwh_capacity
               * nominal_capacity_mwh

The default eol_loss_fraction is 0.20, corresponding to replacement at 80% of
original capacity. Costs per MWh are reported against absolute energy
throughput because the Oxford features use |I * V| integrated over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "model_d"
PRED_COL = f"pred_{MODEL_ID}"


def _find_original_model_coefficients(results: pd.DataFrame, model_id: str) -> dict[str, float]:
    row = results[
        (results["model_id"] == model_id)
        & (results["evaluation"] == "in_sample")
        & (results["scale"] == "original")
    ]
    if row.empty:
        raise ValueError(f"No original-scale in-sample row found for {model_id}")
    row = row.iloc[0]
    def value_or_zero(column: str) -> float:
        value = row.get(column, 0.0)
        if pd.isna(value):
            return 0.0
        return float(value)

    return {
        "throughput_Ah": value_or_zero("coef_throughput_Ah"),
        "sum_current_squared_dt": value_or_zero("coef_sum_current_squared_dt"),
        "high_temp_exposure": value_or_zero("coef_high_temp_exposure"),
        "high_voltage_exposure": value_or_zero("coef_high_voltage_exposure"),
    }


def _initial_capacity_by_cell(predictions: pd.DataFrame) -> pd.Series:
    return (
        predictions.sort_values("interval")
        .groupby(["strategy", "cell"])["capacity_start_Ah"]
        .first()
    )


def build_cost_conversion_table(
    predictions_path: Path,
    results_path: Path,
    replacement_costs_eur_per_mwh_capacity: list[float],
    eol_loss_fraction: float,
    nominal_voltage_v: float,
    model_id: str = MODEL_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build segment-level cost conversion and 15-minute example tables."""
    if eol_loss_fraction <= 0:
        raise ValueError("eol_loss_fraction must be positive")
    if nominal_voltage_v <= 0:
        raise ValueError("nominal_voltage_v must be positive")

    predictions = pd.read_csv(predictions_path)
    results = pd.read_csv(results_path)
    coefs = _find_original_model_coefficients(results, model_id)
    pred_col = f"pred_{model_id}"

    if pred_col not in predictions.columns:
        features = predictions[
            [
                "throughput_Ah",
                "sum_current_squared_dt",
                "high_temp_exposure",
                "high_voltage_exposure",
            ]
        ].fillna(0.0)
        predictions[pred_col] = (
            coefs["throughput_Ah"] * features["throughput_Ah"]
            + coefs["sum_current_squared_dt"] * features["sum_current_squared_dt"]
            + coefs["high_temp_exposure"] * features["high_temp_exposure"]
            + coefs["high_voltage_exposure"] * features["high_voltage_exposure"]
        )

    initial_capacity = _initial_capacity_by_cell(predictions)
    predictions = predictions.copy()
    predictions["initial_capacity_Ah_for_cost"] = [
        initial_capacity.loc[(strategy, cell)]
        for strategy, cell in zip(predictions["strategy"], predictions["cell"])
    ]
    predictions["nominal_capacity_mwh_for_cost"] = (
        predictions["initial_capacity_Ah_for_cost"] * nominal_voltage_v / 1_000_000.0
    )
    predictions["energy_throughput_mwh"] = (
        predictions["approximate_energy_throughput_Wh"] / 1_000_000.0
    )
    predictions["predicted_soh_loss_for_cost"] = predictions[pred_col].clip(lower=0)

    rows: list[pd.DataFrame] = []
    for replacement_cost in replacement_costs_eur_per_mwh_capacity:
        scenario = predictions.copy()
        scenario["replacement_cost_eur_per_mwh_capacity"] = replacement_cost
        scenario["eol_loss_fraction"] = eol_loss_fraction
        scenario["nominal_voltage_v_for_cost"] = nominal_voltage_v
        scenario["degradation_cost_eur"] = (
            scenario["predicted_soh_loss_for_cost"]
            / eol_loss_fraction
            * replacement_cost
            * scenario["nominal_capacity_mwh_for_cost"]
        )
        scenario["degradation_cost_eur_per_mwh_throughput"] = np.where(
            scenario["energy_throughput_mwh"] > 0,
            scenario["degradation_cost_eur"] / scenario["energy_throughput_mwh"],
            np.nan,
        )
        # Oxford throughput is absolute charge + discharge energy. If a market
        # metric is expressed per discharged MWh, a symmetric full-cycle
        # approximation is roughly twice the absolute-throughput value.
        scenario["approx_degradation_cost_eur_per_mwh_discharged"] = (
            2.0 * scenario["degradation_cost_eur_per_mwh_throughput"]
        )
        rows.append(scenario)

    cost_table = pd.concat(rows, ignore_index=True)

    example_table = build_15min_examples(
        coefs=coefs,
        replacement_costs_eur_per_mwh_capacity=replacement_costs_eur_per_mwh_capacity,
        eol_loss_fraction=eol_loss_fraction,
        nominal_voltage_v=nominal_voltage_v,
        nominal_capacity_ah=float(predictions["initial_capacity_Ah_for_cost"].median()),
    )
    return cost_table, example_table


def build_15min_examples(
    coefs: dict[str, float],
    replacement_costs_eur_per_mwh_capacity: list[float],
    eol_loss_fraction: float,
    nominal_voltage_v: float,
    nominal_capacity_ah: float,
) -> pd.DataFrame:
    """Build simple 15-minute stress examples for sanity checking."""
    dt_hours = 0.25
    examples = [
        ("SPM median optimal current, T=25C", 1.4097, 25.0),
        ("SPM mean optimal current, T=25C", 2.613684, 25.0),
        ("Moderate stress 10A, T=25C", 10.0, 25.0),
        ("High stress 20A, T=25C", 20.0, 25.0),
        ("Max SPM optimal step 28.896A, T=25C", 28.896, 25.0),
        ("Max SPM optimal step 28.896A, T=35C", 28.896, 35.0),
    ]

    rows: list[dict[str, float | str]] = []
    nominal_capacity_mwh = nominal_capacity_ah * nominal_voltage_v / 1_000_000.0

    for name, current_a, temp_c in examples:
        throughput_ah = abs(current_a) * dt_hours
        current_squared_dt = current_a**2 * dt_hours
        high_temp_exposure = max(0.0, temp_c - 30.0) * dt_hours
        high_voltage_exposure = 0.0
        predicted_soh_loss = (
            coefs["throughput_Ah"] * throughput_ah
            + coefs["sum_current_squared_dt"] * current_squared_dt
            + coefs["high_temp_exposure"] * high_temp_exposure
            + coefs["high_voltage_exposure"] * high_voltage_exposure
        )
        energy_throughput_mwh = abs(current_a) * nominal_voltage_v * dt_hours / 1_000_000.0

        for replacement_cost in replacement_costs_eur_per_mwh_capacity:
            cost_eur = (
                predicted_soh_loss
                / eol_loss_fraction
                * replacement_cost
                * nominal_capacity_mwh
            )
            eur_per_mwh_throughput = (
                cost_eur / energy_throughput_mwh if energy_throughput_mwh > 0 else np.nan
            )
            rows.append(
                {
                    "case": name,
                    "current_A": current_a,
                    "temperature_C": temp_c,
                    "duration_hours": dt_hours,
                    "throughput_Ah": throughput_ah,
                    "current_squared_A2h": current_squared_dt,
                    "high_temp_degC_h": high_temp_exposure,
                    "predicted_soh_loss_15min": predicted_soh_loss,
                    "energy_throughput_mwh": energy_throughput_mwh,
                    "nominal_capacity_ah_for_cost": nominal_capacity_ah,
                    "nominal_voltage_v_for_cost": nominal_voltage_v,
                    "replacement_cost_eur_per_mwh_capacity": replacement_cost,
                    "eol_loss_fraction": eol_loss_fraction,
                    "degradation_cost_eur": cost_eur,
                    "degradation_cost_eur_per_mwh_throughput": eur_per_mwh_throughput,
                    "approx_degradation_cost_eur_per_mwh_discharged": (
                        2.0 * eur_per_mwh_throughput
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    cost_table: pd.DataFrame,
    example_table: pd.DataFrame,
    replacement_costs: list[float],
    eol_loss_fraction: float,
    nominal_voltage_v: float,
) -> None:
    """Write a concise Markdown report for the cost conversion."""
    base_cost = replacement_costs[0]
    base = cost_table[cost_table["replacement_cost_eur_per_mwh_capacity"] == base_cost]
    by_strategy = (
        base.groupby("strategy")
        .agg(
            segments=("interval", "count"),
            predicted_soh_loss=("predicted_soh_loss_for_cost", "sum"),
            energy_throughput_mwh=("energy_throughput_mwh", "sum"),
            degradation_cost_eur=("degradation_cost_eur", "sum"),
            mean_eur_per_mwh_throughput=(
                "degradation_cost_eur_per_mwh_throughput",
                "mean",
            ),
            mean_eur_per_mwh_discharged=(
                "approx_degradation_cost_eur_per_mwh_discharged",
                "mean",
            ),
        )
        .reset_index()
    )

    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "None.\n"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    values.append("")
                elif isinstance(val, float):
                    values.append(f"{val:.6g}")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"

    report = []
    report.append("# Oxford Degradation Cost Conversion Report\n")
    report.append(
        "This report converts empirical Model D SoH-loss predictions into monetary "
        "degradation-cost estimates. The conversion is for screening and optimizer "
        "penalty calibration, not a market quote or warranty model.\n"
    )
    report.append("## Formula\n")
    report.append(
        "`cost_eur = predicted_soh_loss / eol_loss_fraction * "
        "replacement_cost_eur_per_mwh_capacity * nominal_capacity_mwh`\n"
    )
    report.append(
        "`eur_per_mwh_throughput = cost_eur / absolute_energy_throughput_mwh`\n"
    )
    report.append(
        "Because Oxford energy throughput is absolute charge plus discharge energy, "
        "the report also gives an approximate discharged-MWh value as `2x` the "
        "absolute-throughput value for symmetric full-cycle intuition.\n"
    )
    report.append("## Defaults\n")
    report.append(f"- EOL loss fraction: `{eol_loss_fraction}` (replacement at 80% capacity)")
    report.append(f"- Nominal voltage for Oxford cell energy capacity: `{nominal_voltage_v}` V")
    report.append(
        "- Replacement-cost scenarios in EUR/MWh capacity: "
        + ", ".join(f"{x:.0f}" for x in replacement_costs)
        + "\n"
    )
    report.append("## Base Scenario Summary By Strategy\n")
    report.append(f"Base replacement cost: `{base_cost:.0f}` EUR/MWh capacity.\n")
    report.append(md_table(by_strategy))
    report.append("## 15-Minute Stress Examples\n")
    report.append(md_table(example_table[example_table["replacement_cost_eur_per_mwh_capacity"] == base_cost]))
    report.append("## Market/Planning Context\n")
    report.append(
        "- NREL SAM represents battery degradation through battery-life inputs and can "
        "trigger replacement at a specified capacity threshold, applying replacement "
        "cost in cash flow.\n"
    )
    report.append(
        "- NREL ATB often treats degradation/augmentation through fixed O&M rather "
        "than variable O&M, so explicit EUR/MWh degradation charges are usually an "
        "optimization modeling choice rather than an observed market tariff.\n"
    )
    report.append(
        "- Reviews of BESS optimization literature describe replacement-cost, "
        "energy-throughput, and cycle-counting approaches; 80% remaining capacity "
        "is a common EOL convention.\n"
    )
    report.append("## Limitations\n")
    report.append(
        "- The empirical Model D was fitted at cell/segment level and should be "
        "scaled carefully before use for a full BESS.\n"
    )
    report.append(
        "- Temperature penalties can dominate if assumed cell temperature is above "
        "30 C; use a thermal model or conservative assumption before optimization.\n"
    )
    report.append(
        "- The output is per MWh of absolute battery throughput unless explicitly "
        "using the approximate discharged-MWh column.\n"
    )
    report.append(
        "- This script does not convert to bid strategy directly; it creates a "
        "degradation-cost input for the next optimizer step.\n"
    )
    report.append("## Sources\n")
    report.append("- NREL SAM battery life help: https://samrepo.nlr.gov/help/battery_life.html")
    report.append("- NREL ATB utility-scale battery storage: https://atb.nrel.gov/electricity/2021/utility-scale_battery_storage/")
    report.append("- NREL ATB PV-plus-battery O&M: https://atb.nrel.gov/electricity/2024/2023/utility-scale_pv-plus-battery")
    report.append("- Review of degradation models for UC: https://www.mdpi.com/1996-1073/19/6/1425")
    report.append("- Review of degradation implementation for BESS operation: https://www.mdpi.com/2313-0105/8/9/110")

    path.write_text("\n".join(report))


def _parse_costs(value: str) -> list[float]:
    costs = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not costs:
        raise ValueError("At least one replacement cost must be provided")
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert empirical Oxford degradation predictions to EUR/MWh."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_dataset_with_predictions.csv"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_model_results.csv"),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--replacement-costs-eur-per-mwh-capacity",
        default="60000,100000,150000,300000",
        help="Comma-separated EUR/MWh-capacity replacement cost scenarios.",
    )
    parser.add_argument("--eol-loss-fraction", type=float, default=0.20)
    parser.add_argument("--nominal-voltage-v", type=float, default=3.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_cost_conversion.csv"),
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_cost_15min_examples.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/produced_data/oxford_degradation_cost_conversion_report.md"),
    )
    args = parser.parse_args()

    replacement_costs = _parse_costs(args.replacement_costs_eur_per_mwh_capacity)
    cost_table, examples = build_cost_conversion_table(
        predictions_path=args.predictions,
        results_path=args.results,
        replacement_costs_eur_per_mwh_capacity=replacement_costs,
        eol_loss_fraction=args.eol_loss_fraction,
        nominal_voltage_v=args.nominal_voltage_v,
        model_id=args.model_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cost_table.to_csv(args.output, index=False)
    examples.to_csv(args.examples_output, index=False)
    write_report(
        args.report_output,
        cost_table=cost_table,
        example_table=examples,
        replacement_costs=replacement_costs,
        eol_loss_fraction=args.eol_loss_fraction,
        nominal_voltage_v=args.nominal_voltage_v,
    )

    print(f"Wrote segment cost conversion to {args.output}")
    print(f"Wrote 15-minute examples to {args.examples_output}")
    print(f"Wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
