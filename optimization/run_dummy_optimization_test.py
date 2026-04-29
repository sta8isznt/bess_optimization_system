"""Run a deterministic smoke test for the MILP BESS optimization engine."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FONT_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "bess_optimization_cache"
FONT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(FONT_CACHE_ROOT / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp as pl

from optimization.backtest_utils import (
    build_dummy_cost_curve,
    build_schedule,
    summarize,
    validate_schedule,
)
from optimization.config import params as BASE_PARAMS
from optimization.data import (
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PRICE_SIGNALS_PATH,
    default_2025_dam_file,
    load_degradation_lut_curve,
    load_dam_price_curve,
    load_price_signal_csv_day,
)
from optimization.engine import bess_order


def build_dummy_price_signal(periods: int = 96) -> pd.Series:
    """Build a 24h, 15-minute price signal with clear buy/sell opportunities."""
    index = pd.date_range("2025-01-01", periods=periods, freq="15min")
    hours = index.hour + index.minute / 60.0

    base = 58.0 + 6.0 * np.sin(2.0 * np.pi * (hours - 7.0) / 24.0)
    morning_peak = 35.0 * np.exp(-0.5 * ((hours - 8.0) / 1.7) ** 2)
    evening_peak = 65.0 * np.exp(-0.5 * ((hours - 19.0) / 2.0) ** 2)
    midday_dip = -30.0 * np.exp(-0.5 * ((hours - 13.0) / 2.2) ** 2)
    night_dip = -20.0 * np.exp(-0.5 * ((hours - 3.0) / 1.8) ** 2)

    prices = np.maximum(
        base + morning_peak + evening_peak + midday_dip + night_dip,
        5.0,
    )
    return pd.Series(prices, index=index, name="price_eur_mwh")


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


def write_plot(
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


def write_report(
    summary: dict,
    schedule_path: Path,
    plot_path: Path,
    report_path: Path,
) -> None:
    lines = [
        "# Dummy Optimization Test Report",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-source",
        choices=["price-signals-csv", "dam-2025", "synthetic"],
        default="price-signals-csv",
        help="Price signal to feed into the optimizer.",
    )
    parser.add_argument(
        "--price-signals-csv",
        type=Path,
        default=DEFAULT_PRICE_SIGNALS_PATH,
        help="15-minute DAM CSV file to use with --price-source price-signals-csv.",
    )
    parser.add_argument(
        "--target-date",
        default="2025-01-10",
        help="Date to load from price_signals_15m.csv, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--dam-file",
        type=Path,
        default=None,
        help="Specific Results_2025/DAM Excel file. Defaults to the first 2025 DAM file.",
    )
    parser.add_argument(
        "--degradation-source",
        choices=["lut", "dummy"],
        default="lut",
        help="Use the computed degradation LUT or the old synthetic dummy curve.",
    )
    parser.add_argument(
        "--degradation-lut-csv",
        type=Path,
        default=DEFAULT_DEGRADATION_LUT_PATH,
        help="Degradation lookup table CSV.",
    )
    parser.add_argument(
        "--lut-temperature-c",
        type=float,
        default=25.0,
        help="Temperature slice to use from the degradation LUT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "optimization" / "dummy_outputs",
        help="Directory for the generated CSV, PNG and Markdown report.",
    )
    parser.add_argument("--solver-msg", action="store_true", help="Print solver logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_params = dict(BASE_PARAMS)
    test_params["dt"] = test_params.get("dt", test_params.get("DT", 0.25))
    if args.degradation_source == "lut":
        energy_points, cost_points = load_degradation_lut_curve(
            csv_path=args.degradation_lut_csv,
            temperature_c=args.lut_temperature_c,
        )
        degradation_source = (
            f"{args.degradation_lut_csv.name} at {args.lut_temperature_c:g}C"
        )
    else:
        energy_points, cost_points = build_dummy_cost_curve(test_params)
        degradation_source = "synthetic dummy degradation curve"

    if args.price_source == "price-signals-csv":
        prices = load_price_signal_csv_day(
            csv_path=args.price_signals_csv,
            target_date=args.target_date,
            dt=test_params["dt"],
        )
        price_source_label = (
            f"price_signals_15m.csv ({pd.Timestamp(args.target_date).date()})"
        )
        output_slug = f"price_signals_{pd.Timestamp(args.target_date).strftime('%Y%m%d')}"
    elif args.price_source == "dam-2025":
        dam_file = args.dam_file or default_2025_dam_file()
        prices = load_dam_price_curve(dam_file, dt=test_params["dt"])
        price_source_label = f"DAM 2025 ({dam_file.name})"
        output_slug = dam_file.stem.replace("_EL-DAM_Results_EN_v01", "").lower()
    else:
        prices = build_dummy_price_signal()
        price_source_label = "Synthetic dummy curve"
        output_slug = "synthetic"

    problem, p_buy, p_sell, soc, degradation_cost = bess_order(
        prices=prices.to_numpy(dtype=float),
        battery_params=test_params,
        degradation_energy_points=energy_points,
        degradation_cost_points=cost_points,
        solver_msg=args.solver_msg,
    )
    status_name = pl.LpStatus[problem.status]
    objective_value = float(pl.value(problem.objective) or 0.0)

    schedule = build_schedule(prices, p_buy, p_sell, soc, degradation_cost, test_params)
    summary = summarize(schedule, objective_value, status_name, test_params)
    summary["price_source"] = price_source_label
    summary["price_start"] = str(prices.index.min())
    summary["price_end"] = str(prices.index.max())
    summary["price_min_eur_mwh"] = float(prices.min())
    summary["price_max_eur_mwh"] = float(prices.max())
    summary["price_mean_eur_mwh"] = float(prices.mean())
    summary["degradation_cost_source"] = degradation_source
    validate_schedule(schedule, summary, test_params)

    schedule_path = args.output_dir / f"{output_slug}_optimization_schedule.csv"
    plot_path = args.output_dir / f"{output_slug}_optimization_visual.png"
    report_path = args.output_dir / f"{output_slug}_optimization_report.md"

    schedule.to_csv(schedule_path, index=False)
    write_plot(schedule, summary, plot_path, price_source_label)
    write_report(summary, schedule_path, plot_path, report_path)

    print("Dummy optimization test passed.")
    print(f"Price source: {price_source_label}")
    print(f"Degradation cost source: {degradation_source}")
    print(f"Status: {summary['solver_status']}")
    print(f"Net profit: {summary['net_profit_eur']:.2f} EUR")
    print(f"Schedule CSV: {schedule_path}")
    print(f"Visual output: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
