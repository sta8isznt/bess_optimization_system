"""Daily or annual BESS optimization CLI.

The solve workflow lives in :mod:`bess_optimization.services.optimization`; this
module only keeps the editable defaults and writes local artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bess_optimization.models import OptimizationRequest, OptimizationResult
from bess_optimization.optimization.config import params as BASE_PARAMS
from bess_optimization.paths import (
    ANNUAL_OUTPUT_DIR,
    DAILY_OUTPUT_DIR,
    DEFAULT_PRICE_SIGNALS_PATH,
    DEFAULT_PYBAMM_ONLY_LUT_PATH,
)
from bess_optimization.services.optimization import run_annual_optimization, run_daily_optimization


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_MODE = "daily"  # Choose "daily" or "annual".
TARGET_DATE = "2025-11-01"
YEAR = 2025
DEGRADATION_SOURCE = "pybamm_only"
LUT_TEMPERATURE_C = 25.0
PRICE_SIGNALS_CSV = DEFAULT_PRICE_SIGNALS_PATH
DEGRADATION_LUT_CSV = DEFAULT_PYBAMM_ONLY_LUT_PATH
SOLVER_MSG = False
INSTALLED_CAPACITY_MW = 50.0


def build_params() -> dict:
    params = dict(BASE_PARAMS)
    params["dt"] = params.get("dt", params.get("DT", 0.25))
    params["terminal_soc_mode"] = "equal_initial"
    return params


def _configure_matplotlib_cache() -> None:
    font_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "bess_optimization_cache"
    font_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(font_cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_root / "xdg"))


def write_daily_plot(schedule: pd.DataFrame, summary: dict, output_path: Path) -> None:
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bar_width_days = 0.25 / 24.0 * 0.86
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    colors = {"buy": "#2ca02c", "sell": "#d62728"}
    for row in schedule.itertuples(index=False):
        if row.operation in colors:
            axes[0].axvspan(
                row.timestamp,
                row.timestamp + pd.Timedelta(minutes=15),
                color=colors[row.operation],
                alpha=0.16,
            )

    axes[0].plot(schedule["timestamp"], schedule["price_eur_mwh"], color="#1f2933", linewidth=1.8)
    axes[0].set_ylabel("EUR/MWh")
    axes[0].grid(alpha=0.25)
    axes[1].bar(schedule["timestamp"], schedule["p_sell_mw"], width=bar_width_days, color="#d62728", label="Sell MW")
    axes[1].bar(schedule["timestamp"], -schedule["p_buy_mw"], width=bar_width_days, color="#2ca02c", label="Buy MW")
    axes[1].axhline(0.0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("Power MW")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    axes[2].plot(schedule["timestamp"], schedule["soc_pct"] * 100.0, color="#2563eb", linewidth=1.8)
    axes[2].set_ylabel("SoC %")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.25)
    fig.suptitle(
        f"Status: {summary['solver_status']}    "
        f"Net profit: {summary['net_profit_eur']:.2f} EUR    "
        f"Park net profit: {summary['park_net_profit_eur']:.2f} EUR",
        fontsize=11,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(result: OptimizationResult, report_path: Path, output_paths: dict[str, Path]) -> None:
    summary = result.summary_dict
    lines = [
        "# Battery Optimization Backtest",
        "",
        "## Outputs",
    ]
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path.name}`")
    lines.extend(["", "## Summary"])
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_daily_outputs(result: OptimizationResult) -> dict[str, Path]:
    DAILY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_day = pd.Timestamp(result.summary_dict["date"])
    output_slug = f"daily_{target_day.strftime('%Y%m%d')}"
    paths = {
        "Schedule CSV": DAILY_OUTPUT_DIR / f"{output_slug}_schedule.csv",
        "Visual output": DAILY_OUTPUT_DIR / f"{output_slug}_visual.png",
        "Report": DAILY_OUTPUT_DIR / f"{output_slug}_report.md",
    }
    result.dispatch_df.to_csv(paths["Schedule CSV"], index=False)
    write_daily_plot(result.dispatch_df, result.summary_dict, paths["Visual output"])
    if result.benchmark_comparison_df is not None:
        paths["Benchmark comparison CSV"] = DAILY_OUTPUT_DIR / f"{output_slug}_benchmark_comparison.csv"
        result.benchmark_comparison_df.to_csv(paths["Benchmark comparison CSV"], index=False)
    write_report(result, paths["Report"], paths)
    return paths


def write_annual_outputs(result: OptimizationResult) -> dict[str, Path]:
    ANNUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year = int(result.summary_dict["year"])
    paths = {
        "Schedule CSV": ANNUAL_OUTPUT_DIR / f"annual_{year}_schedule.csv",
        "Daily stats CSV": ANNUAL_OUTPUT_DIR / f"annual_{year}_daily_stats.csv",
        "Monthly stats CSV": ANNUAL_OUTPUT_DIR / f"annual_{year}_monthly_stats.csv",
        "Report": ANNUAL_OUTPUT_DIR / f"annual_{year}_report.md",
    }
    result.dispatch_df.to_csv(paths["Schedule CSV"], index=False)
    if result.daily_stats_df is not None:
        result.daily_stats_df.to_csv(paths["Daily stats CSV"], index=False)
    monthly = result.dispatch_df.copy()
    monthly["month"] = pd.to_datetime(monthly["timestamp"]).dt.to_period("M").astype(str)
    monthly_stats = (
        monthly.groupby("month")
        .agg(
            intervals=("timestamp", "count"),
            gross_revenue_eur=("gross_revenue_eur", "sum"),
            gross_purchase_eur=("gross_purchase_eur", "sum"),
            degradation_cost_eur=("degradation_cost_eur", "sum"),
            net_profit_eur=("interval_profit_eur", "sum"),
            buy_energy_mwh=("buy_energy_mwh", "sum"),
            sell_energy_mwh=("sell_energy_mwh", "sum"),
            price_min_eur_mwh=("price_eur_mwh", "min"),
            price_max_eur_mwh=("price_eur_mwh", "max"),
            price_mean_eur_mwh=("price_eur_mwh", "mean"),
        )
        .reset_index()
    )
    monthly_stats.to_csv(paths["Monthly stats CSV"], index=False)
    if result.benchmark_comparison_df is not None:
        paths["Benchmark comparison CSV"] = ANNUAL_OUTPUT_DIR / f"annual_{year}_benchmark_comparison.csv"
        result.benchmark_comparison_df.to_csv(paths["Benchmark comparison CSV"], index=False)
    write_report(result, paths["Report"], paths)
    return paths


def main() -> None:
    request = OptimizationRequest(
        run_mode=RUN_MODE,
        target_date=TARGET_DATE,
        year=YEAR,
        price_file=PRICE_SIGNALS_CSV,
        degradation_source=DEGRADATION_SOURCE,
        degradation_lut_file=DEGRADATION_LUT_CSV,
        temperature_c=LUT_TEMPERATURE_C,
        params_override=build_params(),
        scale_capacity_mw=INSTALLED_CAPACITY_MW,
        solver_msg=SOLVER_MSG,
    )
    mode = RUN_MODE.strip().lower()
    if mode == "daily":
        result = run_daily_optimization(request)
        paths = write_daily_outputs(result)
        label = f"Date: {pd.Timestamp(TARGET_DATE).date()}"
    elif mode == "annual":
        result = run_annual_optimization(request)
        paths = write_annual_outputs(result)
        label = f"Year: {YEAR}"
    else:
        raise ValueError('RUN_MODE must be "daily" or "annual".')

    summary = result.summary_dict
    print(f"{mode.title()} optimization completed.")
    print(label)
    print(f"Degradation cost source: {summary['degradation_cost_source']}")
    print(f"Status: {summary['solver_status']}")
    print(f"Base-module net profit: {summary['net_profit_eur']:.2f} EUR")
    print(f"Park net profit: {summary['park_net_profit_eur']:.2f} EUR")
    for output_label, path in paths.items():
        print(f"{output_label}: {path}")


if __name__ == "__main__":
    main()
