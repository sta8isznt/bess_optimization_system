"""Run daily or annual BESS optimization by editing the settings below."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FONT_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "bess_optimization_cache"
FONT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FONT_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pulp as pl

from optimization.backtest_utils import (
    build_dummy_cost_curve,
    build_schedule,
    safe_divide,
    summarize,
    validate_schedule,
)
from optimization.backtesting_loaders import (
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PYBAMM_ONLY_LUT_PATH,
    DEFAULT_PRICE_SIGNALS_PATH,
    load_degradation_lut_curve,
    load_price_signal_day,
    load_price_signal_year,
)
from optimization.config import params as BASE_PARAMS
from optimization.engine import bess_order


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_MODE = "daily"  # Choose "daily" or "annual".

TARGET_DATE = "2025-11-01"  # Used when RUN_MODE = "daily".
YEAR = 2025  # Used when RUN_MODE = "annual".

DEGRADATION_SOURCE = "pybamm_only"  # Choose "lut", "pybamm_only", or "dummy".
LUT_TEMPERATURE_C = 25.0

PRICE_SIGNALS_CSV = DEFAULT_PRICE_SIGNALS_PATH
DEGRADATION_LUT_CSV = DEFAULT_DEGRADATION_LUT_PATH
PYBAMM_ONLY_LUT_CSV = DEFAULT_PYBAMM_ONLY_LUT_PATH

SOLVER_MSG = False
PROGRESS_EVERY = 25  # Used only for annual runs. Set 0 to disable progress prints.

INSTALLED_CAPACITY_MW = 50.0
BASE_MODULE_MW = float(BASE_PARAMS["p_max"])
SCALE_FACTOR = INSTALLED_CAPACITY_MW / BASE_MODULE_MW

DAILY_OUTPUT_DIR = BASE_DIR / "daily_outputs"
ANNUAL_OUTPUT_DIR = BASE_DIR / "annual_outputs"


def build_test_params() -> dict:
    test_params = dict(BASE_PARAMS)
    test_params["dt"] = test_params.get("dt", test_params.get("DT", 0.25))
    return test_params


def load_degradation_curve(test_params: dict) -> tuple[list[float], list[float], str]:
    if DEGRADATION_SOURCE == "lut":
        energy_points, cost_points = load_degradation_lut_curve(
            csv_path=DEGRADATION_LUT_CSV,
            temperature_c=LUT_TEMPERATURE_C,
        )
        source_label = f"{DEGRADATION_LUT_CSV.name} at {LUT_TEMPERATURE_C:g}C"
        return energy_points, cost_points, source_label

    if DEGRADATION_SOURCE == "pybamm_only":
        energy_points, cost_points = load_degradation_lut_curve(
            csv_path=PYBAMM_ONLY_LUT_CSV,
            temperature_c=LUT_TEMPERATURE_C,
        )
        source_label = f"{PYBAMM_ONLY_LUT_CSV.name} at {LUT_TEMPERATURE_C:g}C"
        return energy_points, cost_points, source_label

    if DEGRADATION_SOURCE == "dummy":
        energy_points, cost_points = build_dummy_cost_curve(test_params)
        return energy_points, cost_points, "synthetic dummy degradation curve"

    raise ValueError('DEGRADATION_SOURCE must be "lut", "pybamm_only", or "dummy".')


def validate_scale_settings(test_params: dict) -> float:
    if BASE_MODULE_MW <= 0:
        raise ValueError("BASE_MODULE_MW must be positive for park scale-up.")
    if INSTALLED_CAPACITY_MW <= 0:
        raise ValueError("INSTALLED_CAPACITY_MW must be positive for park scale-up.")

    configured_base_mw = float(test_params["p_max"])
    if abs(configured_base_mw - BASE_MODULE_MW) > 1e-9:
        raise ValueError(
            "BASE_MODULE_MW must match test_params['p_max'] so post-processing "
            "scale-up stays aligned with the solved base module."
        )
    return float(SCALE_FACTOR)


def add_park_scale_up_columns(schedule: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    scaled = schedule.copy()
    columns_to_scale = {
        "p_buy_mw": "park_p_buy_mw",
        "p_sell_mw": "park_p_sell_mw",
        "net_export_mw": "park_net_export_mw",
        "buy_energy_mwh": "park_buy_energy_mwh",
        "sell_energy_mwh": "park_sell_energy_mwh",
        "soc_mwh": "park_soc_mwh",
        "gross_revenue_eur": "park_gross_revenue_eur",
        "gross_purchase_eur": "park_gross_purchase_eur",
        "degradation_cost_eur": "park_degradation_cost_eur",
        "interval_profit_eur": "park_interval_profit_eur",
    }
    for source_col, target_col in columns_to_scale.items():
        scaled[target_col] = scaled[source_col] * scale_factor
    return scaled


def add_park_scale_up_summary(summary: dict, test_params: dict, scale_factor: float) -> dict:
    out = dict(summary)
    out["installed_capacity_mw"] = float(INSTALLED_CAPACITY_MW)
    out["installed_energy_capacity_mwh"] = float(test_params["e_max"] * scale_factor)
    out["scale_factor"] = float(scale_factor)
    out["park_net_profit_eur"] = float(out["net_profit_eur"] * scale_factor)
    out["park_gross_revenue_eur"] = float(out["gross_revenue_eur"] * scale_factor)
    out["park_gross_purchase_eur"] = float(out["gross_purchase_eur"] * scale_factor)
    out["park_degradation_cost_eur"] = float(out["degradation_cost_eur"] * scale_factor)
    out["park_buy_energy_mwh"] = float(out["buy_energy_mwh"] * scale_factor)
    out["park_sell_energy_mwh"] = float(out["sell_energy_mwh"] * scale_factor)
    return out


def add_park_scale_up_metadata(stats: pd.DataFrame, test_params: dict) -> pd.DataFrame:
    scale_factor = validate_scale_settings(test_params)
    stats = stats.copy()
    stats["installed_capacity_mw"] = float(INSTALLED_CAPACITY_MW)
    stats["installed_energy_capacity_mwh"] = float(test_params["e_max"] * scale_factor)
    stats["scale_factor"] = float(scale_factor)
    return stats


def optimize_prices(
    prices: pd.Series,
    test_params: dict,
    energy_points: list[float],
    cost_points: list[float],
) -> tuple[pd.DataFrame, dict]:
    scale_factor = validate_scale_settings(test_params)
    problem, p_buy, p_sell, soc, degradation_cost = bess_order(
        prices=prices.to_numpy(dtype=float),
        battery_params=test_params,
        degradation_energy_points=energy_points,
        degradation_cost_points=cost_points,
        solver_msg=SOLVER_MSG,
    )
    status_name = pl.LpStatus[problem.status]
    objective_value = float(pl.value(problem.objective) or 0.0)

    schedule = build_schedule(
        prices=prices,
        p_buy=p_buy,
        p_sell=p_sell,
        soc=soc,
        degradation_cost=degradation_cost,
        test_params=test_params,
    )
    summary = summarize(schedule, objective_value, status_name, test_params)
    summary["date"] = prices.index.min().date().isoformat()
    summary["price_min_eur_mwh"] = float(prices.min())
    summary["price_max_eur_mwh"] = float(prices.max())
    summary["price_mean_eur_mwh"] = float(prices.mean())
    validate_schedule(schedule, summary, test_params)
    schedule = add_park_scale_up_columns(schedule, scale_factor)
    summary = add_park_scale_up_summary(summary, test_params, scale_factor)
    return schedule, summary


def shade_operations(
    ax: plt.Axes,
    schedule: pd.DataFrame,
    interval_minutes: int = 15,
) -> None:
    labels_seen = set()
    interval = pd.Timedelta(minutes=interval_minutes)
    colors = {"sell": "#d62728", "buy": "#2ca02c"}
    labels = {"sell": "Sell energy", "buy": "Buy energy"}

    for row in schedule.itertuples(index=False):
        if row.operation not in colors:
            continue
        label = labels[row.operation] if row.operation not in labels_seen else None
        ax.axvspan(
            row.timestamp,
            row.timestamp + interval,
            color=colors[row.operation],
            alpha=0.16,
            label=label,
        )
        labels_seen.add(row.operation)


def write_daily_plot(
    schedule: pd.DataFrame,
    summary: dict,
    output_path: Path,
    price_source_label: str,
) -> None:
    bar_width_days = 0.25 / 24.0 * 0.86
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    shade_operations(axes[0], schedule)
    axes[0].plot(
        schedule["timestamp"],
        schedule["price_eur_mwh"],
        color="#1f2933",
        linewidth=1.8,
    )
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title(f"BESS optimization price signal - {price_source_label}")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        schedule["timestamp"],
        schedule["p_sell_mw"],
        width=bar_width_days,
        color="#d62728",
        label="Sell MW",
    )
    axes[1].bar(
        schedule["timestamp"],
        -schedule["p_buy_mw"],
        width=bar_width_days,
        color="#2ca02c",
        label="Buy MW",
    )
    axes[1].axhline(0.0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("Power MW")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        schedule["timestamp"],
        schedule["soc_pct"] * 100.0,
        color="#2563eb",
        linewidth=1.8,
    )
    axes[2].set_ylabel("SoC %")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.25)

    stats = (
        f"Status: {summary['solver_status']}    "
        f"Net profit: {summary['net_profit_eur']:.2f} EUR    "
        f"Bought: {summary['buy_energy_mwh']:.3f} MWh    "
        f"Sold: {summary['sell_energy_mwh']:.3f} MWh    "
        f"Degradation: {summary['degradation_cost_eur']:.2f} EUR"
    )
    fig.suptitle(stats, fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_daily_report(
    summary: dict,
    schedule_path: Path,
    plot_path: Path,
    report_path: Path,
) -> None:
    lines = [
        "# Daily Battery Optimization Backtest",
        "",
        "## Outputs",
        f"- Schedule CSV: `{schedule_path.name}`",
        f"- Visual output: `{plot_path.name}`",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def add_unit_metrics(stats: pd.DataFrame, test_params: dict) -> pd.DataFrame:
    stats = stats.copy()
    stats["gross_revenue_eur_per_mwh_sold"] = stats.apply(
        lambda row: safe_divide(row["gross_revenue_eur"], row["sell_energy_mwh"]),
        axis=1,
    )
    stats["gross_purchase_eur_per_mwh_bought"] = stats.apply(
        lambda row: safe_divide(row["gross_purchase_eur"], row["buy_energy_mwh"]),
        axis=1,
    )
    stats["net_profit_eur_per_mwh_sold"] = stats.apply(
        lambda row: safe_divide(row["net_profit_eur"], row["sell_energy_mwh"]),
        axis=1,
    )
    stats["degradation_cost_eur_per_mwh_sold"] = stats.apply(
        lambda row: safe_divide(row["degradation_cost_eur"], row["sell_energy_mwh"]),
        axis=1,
    )
    stats["equivalent_discharge_cycles"] = stats["sell_energy_mwh"].apply(
        lambda value: safe_divide(value, test_params["e_max"])
    )
    stats["net_profit_eur_per_mwh_capacity"] = stats["net_profit_eur"].apply(
        lambda value: safe_divide(value, test_params["e_max"])
    )
    return stats


def build_monthly_stats(schedule: pd.DataFrame, test_params: dict) -> pd.DataFrame:
    work = schedule.copy()
    work["month"] = work["timestamp"].dt.to_period("M").astype(str)
    aggregations = {
        "intervals": ("timestamp", "count"),
        "gross_revenue_eur": ("gross_revenue_eur", "sum"),
        "gross_purchase_eur": ("gross_purchase_eur", "sum"),
        "degradation_cost_eur": ("degradation_cost_eur", "sum"),
        "net_profit_eur": ("interval_profit_eur", "sum"),
        "buy_energy_mwh": ("buy_energy_mwh", "sum"),
        "sell_energy_mwh": ("sell_energy_mwh", "sum"),
        "buy_intervals": ("operation", lambda values: int((values == "buy").sum())),
        "sell_intervals": ("operation", lambda values: int((values == "sell").sum())),
        "idle_intervals": ("operation", lambda values: int((values == "idle").sum())),
        "price_min_eur_mwh": ("price_eur_mwh", "min"),
        "price_max_eur_mwh": ("price_eur_mwh", "max"),
        "price_mean_eur_mwh": ("price_eur_mwh", "mean"),
    }
    park_aggregations = {
        "park_gross_revenue_eur": ("park_gross_revenue_eur", "sum"),
        "park_gross_purchase_eur": ("park_gross_purchase_eur", "sum"),
        "park_degradation_cost_eur": ("park_degradation_cost_eur", "sum"),
        "park_net_profit_eur": ("park_interval_profit_eur", "sum"),
        "park_buy_energy_mwh": ("park_buy_energy_mwh", "sum"),
        "park_sell_energy_mwh": ("park_sell_energy_mwh", "sum"),
    }
    for output_col, aggregation in park_aggregations.items():
        if aggregation[0] in work.columns:
            aggregations[output_col] = aggregation

    monthly = (
        work.groupby("month")
        .agg(**aggregations)
        .reset_index()
    )
    monthly = add_unit_metrics(monthly, test_params)
    return add_park_scale_up_metadata(monthly, test_params)


def build_annual_summary(
    schedule: pd.DataFrame,
    daily_stats: pd.DataFrame,
    test_params: dict,
    degradation_source: str,
) -> dict:
    gross_revenue = float(schedule["gross_revenue_eur"].sum())
    gross_purchase = float(schedule["gross_purchase_eur"].sum())
    degradation_cost = float(schedule["degradation_cost_eur"].sum())
    net_profit = float(schedule["interval_profit_eur"].sum())
    buy_energy = float(schedule["buy_energy_mwh"].sum())
    sell_energy = float(schedule["sell_energy_mwh"].sum())

    summary = {
        "year": YEAR,
        "days": int(daily_stats.shape[0]),
        "intervals": int(schedule.shape[0]),
        "optimal_days": int((daily_stats["solver_status"] == "Optimal").sum()),
        "gross_revenue_eur": gross_revenue,
        "gross_purchase_eur": gross_purchase,
        "degradation_cost_eur": degradation_cost,
        "net_profit_eur": net_profit,
        "buy_energy_mwh": buy_energy,
        "sell_energy_mwh": sell_energy,
        "equivalent_discharge_cycles": safe_divide(sell_energy, test_params["e_max"]),
        "gross_revenue_eur_per_mwh_sold": safe_divide(gross_revenue, sell_energy),
        "gross_purchase_eur_per_mwh_bought": safe_divide(gross_purchase, buy_energy),
        "net_profit_eur_per_mwh_sold": safe_divide(net_profit, sell_energy),
        "degradation_cost_eur_per_mwh_sold": safe_divide(degradation_cost, sell_energy),
        "net_profit_eur_per_mwh_capacity_year": safe_divide(
            net_profit,
            test_params["e_max"],
        ),
        "buy_intervals": int((schedule["operation"] == "buy").sum()),
        "sell_intervals": int((schedule["operation"] == "sell").sum()),
        "idle_intervals": int((schedule["operation"] == "idle").sum()),
        "price_min_eur_mwh": float(schedule["price_eur_mwh"].min()),
        "price_max_eur_mwh": float(schedule["price_eur_mwh"].max()),
        "price_mean_eur_mwh": float(schedule["price_eur_mwh"].mean()),
        "battery_power_mw": float(test_params["p_max"]),
        "battery_capacity_mwh": float(test_params["e_max"]),
        "degradation_cost_source": degradation_source,
    }
    return add_park_scale_up_summary(
        summary,
        test_params,
        validate_scale_settings(test_params),
    )


def write_annual_report(
    summary: dict,
    schedule_path: Path,
    daily_path: Path,
    monthly_path: Path,
    report_path: Path,
) -> None:
    lines = [
        f"# Annual Battery Optimization Backtest {summary['year']}",
        "",
        "## Key Metrics",
        (
            "- Installed park: "
            f"{summary['installed_capacity_mw']:.3f} MW / "
            f"{summary['installed_energy_capacity_mwh']:.3f} MWh "
            f"(scale factor {summary['scale_factor']:.3f})"
        ),
        f"- Park net profit: {summary['park_net_profit_eur']:.2f} EUR",
        f"- Park gross revenue: {summary['park_gross_revenue_eur']:.2f} EUR",
        f"- Park gross purchase cost: {summary['park_gross_purchase_eur']:.2f} EUR",
        f"- Park degradation cost: {summary['park_degradation_cost_eur']:.2f} EUR",
        f"- Park sold energy: {summary['park_sell_energy_mwh']:.3f} MWh",
        f"- Park bought energy: {summary['park_buy_energy_mwh']:.3f} MWh",
        f"- Base-module net profit: {summary['net_profit_eur']:.2f} EUR",
        f"- Base-module sold energy: {summary['sell_energy_mwh']:.3f} MWh",
        (
            "- Gross revenue per MWh sold: "
            f"{summary['gross_revenue_eur_per_mwh_sold']:.2f} EUR/MWh"
        ),
        (
            "- Net profit per MWh sold: "
            f"{summary['net_profit_eur_per_mwh_sold']:.2f} EUR/MWh"
        ),
        (
            "- Net profit per MWh battery capacity-year: "
            f"{summary['net_profit_eur_per_mwh_capacity_year']:.2f} EUR/MWh-year"
        ),
        "",
        "## Outputs",
        f"- Full schedule CSV: `{schedule_path.name}`",
        f"- Daily stats CSV: `{daily_path.name}`",
        f"- Monthly stats CSV: `{monthly_path.name}`",
        "",
        "## Full Summary",
    ]

    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_daily_optimization() -> None:
    output_dir = DAILY_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    test_params = build_test_params()
    energy_points, cost_points, degradation_source = load_degradation_curve(test_params)

    prices = load_price_signal_day(
        csv_path=PRICE_SIGNALS_CSV,
        target_date=TARGET_DATE,
        dt=test_params["dt"],
    )
    schedule, summary = optimize_prices(
        prices=prices,
        test_params=test_params,
        energy_points=energy_points,
        cost_points=cost_points,
    )

    target_day = pd.Timestamp(TARGET_DATE).date()
    price_source_label = f"{PRICE_SIGNALS_CSV.name} ({target_day})"
    output_slug = f"daily_{pd.Timestamp(TARGET_DATE).strftime('%Y%m%d')}"

    summary["price_source"] = price_source_label
    summary["price_start"] = str(prices.index.min())
    summary["price_end"] = str(prices.index.max())
    summary["degradation_cost_source"] = degradation_source

    schedule_path = output_dir / f"{output_slug}_schedule.csv"
    plot_path = output_dir / f"{output_slug}_visual.png"
    report_path = output_dir / f"{output_slug}_report.md"

    schedule.to_csv(schedule_path, index=False)
    write_daily_plot(schedule, summary, plot_path, price_source_label)
    write_daily_report(summary, schedule_path, plot_path, report_path)

    print("Daily optimization completed.")
    print(f"Date: {target_day}")
    print(f"Degradation cost source: {degradation_source}")
    print(f"Status: {summary['solver_status']}")
    print(
        "Installed park: "
        f"{summary['installed_capacity_mw']:.3f} MW / "
        f"{summary['installed_energy_capacity_mwh']:.3f} MWh "
        f"(scale factor {summary['scale_factor']:.3f})"
    )
    print(f"Base-module net profit: {summary['net_profit_eur']:.2f} EUR")
    print(f"Park net profit: {summary['park_net_profit_eur']:.2f} EUR")
    print(f"Schedule CSV: {schedule_path}")
    print(f"Visual output: {plot_path}")
    print(f"Report: {report_path}")


def run_annual_optimization() -> None:
    output_dir = ANNUAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    test_params = build_test_params()
    energy_points, cost_points, degradation_source = load_degradation_curve(test_params)

    prices_year = load_price_signal_year(
        csv_path=PRICE_SIGNALS_CSV,
        year=YEAR,
        dt=test_params["dt"],
    )
    grouped_days = list(prices_year.groupby(prices_year.index.normalize()))

    schedules = []
    daily_summaries = []
    for day_index, (day_start, day_prices) in enumerate(grouped_days, start=1):
        day_schedule, day_summary = optimize_prices(
            prices=day_prices,
            test_params=test_params,
            energy_points=energy_points,
            cost_points=cost_points,
        )
        schedules.append(day_schedule)
        daily_summaries.append(day_summary)

        if PROGRESS_EVERY and day_index % PROGRESS_EVERY == 0:
            print(
                f"Optimized {day_index}/{len(grouped_days)} days "
                f"through {day_start.date()}..."
            )

    annual_schedule = pd.concat(schedules, ignore_index=True)
    daily_stats = pd.DataFrame(daily_summaries)
    monthly_stats = build_monthly_stats(annual_schedule, test_params)
    annual_summary = build_annual_summary(
        schedule=annual_schedule,
        daily_stats=daily_stats,
        test_params=test_params,
        degradation_source=degradation_source,
    )

    schedule_path = output_dir / f"annual_{YEAR}_schedule.csv"
    daily_path = output_dir / f"annual_{YEAR}_daily_stats.csv"
    monthly_path = output_dir / f"annual_{YEAR}_monthly_stats.csv"
    report_path = output_dir / f"annual_{YEAR}_report.md"

    annual_schedule.to_csv(schedule_path, index=False)
    daily_stats.to_csv(daily_path, index=False)
    monthly_stats.to_csv(monthly_path, index=False)
    write_annual_report(annual_summary, schedule_path, daily_path, monthly_path, report_path)

    print("Annual optimization completed.")
    print(f"Year: {YEAR}")
    print(f"Days optimized: {annual_summary['days']}")
    print(f"Degradation cost source: {annual_summary['degradation_cost_source']}")
    print(
        "Installed park: "
        f"{annual_summary['installed_capacity_mw']:.3f} MW / "
        f"{annual_summary['installed_energy_capacity_mwh']:.3f} MWh "
        f"(scale factor {annual_summary['scale_factor']:.3f})"
    )
    print(f"Base-module net profit: {annual_summary['net_profit_eur']:.2f} EUR")
    print(f"Park net profit: {annual_summary['park_net_profit_eur']:.2f} EUR")
    print(
        "Gross revenue per MWh sold: "
        f"{annual_summary['gross_revenue_eur_per_mwh_sold']:.2f} EUR/MWh"
    )
    print(
        "Net profit per MWh sold: "
        f"{annual_summary['net_profit_eur_per_mwh_sold']:.2f} EUR/MWh"
    )
    print(f"Report: {report_path}")


def main() -> None:
    mode = RUN_MODE.strip().lower()
    if mode == "daily":
        run_daily_optimization()
    elif mode == "annual":
        run_annual_optimization()
    else:
        raise ValueError('RUN_MODE must be "daily" or "annual".')


if __name__ == "__main__":
    main()
