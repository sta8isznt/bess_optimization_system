"""Run a full-year BESS optimization backtest using 15-minute DAM prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pulp as pl

from optimization.backtest_utils import (
    build_dummy_cost_curve,
    build_schedule,
    safe_divide,
    summarize,
    validate_schedule,
)
from optimization.config import params as BASE_PARAMS
from optimization.data import (
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PRICE_SIGNALS_PATH,
    load_degradation_lut_curve,
    load_price_signal_csv_year,
)
from optimization.engine import bess_order


def optimize_day(
    prices: pd.Series,
    test_params: dict,
    energy_points: list[float],
    cost_points: list[float],
    solver_msg: bool,
) -> tuple[pd.DataFrame, dict]:
    problem, p_buy, p_sell, soc, degradation_cost = bess_order(
        prices=prices.to_numpy(dtype=float),
        battery_params=test_params,
        degradation_energy_points=energy_points,
        degradation_cost_points=cost_points,
        solver_msg=solver_msg,
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
    return schedule, summary


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
    monthly = (
        work.groupby("month")
        .agg(
            intervals=("timestamp", "count"),
            gross_revenue_eur=("gross_revenue_eur", "sum"),
            gross_purchase_eur=("gross_purchase_eur", "sum"),
            degradation_cost_eur=("degradation_cost_eur", "sum"),
            net_profit_eur=("interval_profit_eur", "sum"),
            buy_energy_mwh=("buy_energy_mwh", "sum"),
            sell_energy_mwh=("sell_energy_mwh", "sum"),
            buy_intervals=("operation", lambda values: int((values == "buy").sum())),
            sell_intervals=("operation", lambda values: int((values == "sell").sum())),
            idle_intervals=("operation", lambda values: int((values == "idle").sum())),
            price_min_eur_mwh=("price_eur_mwh", "min"),
            price_max_eur_mwh=("price_eur_mwh", "max"),
            price_mean_eur_mwh=("price_eur_mwh", "mean"),
        )
        .reset_index()
    )
    return add_unit_metrics(monthly, test_params)


def build_annual_summary(
    schedule: pd.DataFrame,
    daily_stats: pd.DataFrame,
    test_params: dict,
    year: int,
    degradation_source: str,
) -> dict:
    gross_revenue = float(schedule["gross_revenue_eur"].sum())
    gross_purchase = float(schedule["gross_purchase_eur"].sum())
    degradation_cost = float(schedule["degradation_cost_eur"].sum())
    net_profit = float(schedule["interval_profit_eur"].sum())
    buy_energy = float(schedule["buy_energy_mwh"].sum())
    sell_energy = float(schedule["sell_energy_mwh"].sum())

    return {
        "year": year,
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


def write_report(
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
        f"- Net profit: {summary['net_profit_eur']:.2f} EUR",
        f"- Gross revenue: {summary['gross_revenue_eur']:.2f} EUR",
        f"- Gross purchase cost: {summary['gross_purchase_eur']:.2f} EUR",
        f"- Degradation cost: {summary['degradation_cost_eur']:.2f} EUR",
        f"- Sold energy: {summary['sell_energy_mwh']:.3f} MWh",
        f"- Bought energy: {summary['buy_energy_mwh']:.3f} MWh",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--price-signals-csv",
        type=Path,
        default=DEFAULT_PRICE_SIGNALS_PATH,
        help="15-minute DAM price CSV.",
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
        default=REPO_ROOT / "optimization" / "annual_outputs",
    )
    parser.add_argument("--solver-msg", action="store_true", help="Print solver logs.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N optimized days. Use 0 to disable.",
    )
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

    prices_year = load_price_signal_csv_year(
        csv_path=args.price_signals_csv,
        year=args.year,
        dt=test_params["dt"],
    )
    grouped_days = list(prices_year.groupby(prices_year.index.normalize()))

    schedules = []
    daily_summaries = []
    for day_index, (day_start, day_prices) in enumerate(grouped_days, start=1):
        day_schedule, day_summary = optimize_day(
            prices=day_prices,
            test_params=test_params,
            energy_points=energy_points,
            cost_points=cost_points,
            solver_msg=args.solver_msg,
        )
        schedules.append(day_schedule)
        daily_summaries.append(day_summary)

        if args.progress_every and day_index % args.progress_every == 0:
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
        year=args.year,
        degradation_source=degradation_source,
    )

    schedule_path = args.output_dir / f"annual_{args.year}_schedule.csv"
    daily_path = args.output_dir / f"annual_{args.year}_daily_stats.csv"
    monthly_path = args.output_dir / f"annual_{args.year}_monthly_stats.csv"
    report_path = args.output_dir / f"annual_{args.year}_report.md"

    annual_schedule.to_csv(schedule_path, index=False)
    daily_stats.to_csv(daily_path, index=False)
    monthly_stats.to_csv(monthly_path, index=False)
    write_report(annual_summary, schedule_path, daily_path, monthly_path, report_path)

    print("Annual optimization backtest passed.")
    print(f"Year: {args.year}")
    print(f"Days optimized: {annual_summary['days']}")
    print(f"Degradation cost source: {annual_summary['degradation_cost_source']}")
    print(f"Net profit: {annual_summary['net_profit_eur']:.2f} EUR")
    print(
        "Gross revenue per MWh sold: "
        f"{annual_summary['gross_revenue_eur_per_mwh_sold']:.2f} EUR/MWh"
    )
    print(
        "Net profit per MWh sold: "
        f"{annual_summary['net_profit_eur_per_mwh_sold']:.2f} EUR/MWh"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
