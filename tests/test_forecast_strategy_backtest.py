from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from optimization.forecast_strategy_backtest import (
    build_test_params,
    build_trade_ledger,
    forecast_series_for_day,
    run_forecast_strategy_backtest,
    settle_schedule_on_actual_prices,
)
from optimization.forecasting.dam_15min_forecast import build_synthetic_history


class ForecastStrategyBacktestTests(unittest.TestCase):
    def test_forecast_for_day_does_not_use_target_day_actuals(self) -> None:
        history = build_synthetic_history(days=45, start_date="2025-11-01")
        target_day = pd.Timestamp("2025-12-10")

        baseline, _, _ = forecast_series_for_day(
            history,
            target_day=target_day,
            window_days=30,
        )

        changed = history.copy()
        mask = (
            pd.to_datetime(changed["timestamp"]).dt.normalize()
            == target_day.normalize()
        )
        changed.loc[mask, "price_eur_mwh"] = 10_000.0
        mutated, _, _ = forecast_series_for_day(
            changed,
            target_day=target_day,
            window_days=30,
        )

        np.testing.assert_allclose(baseline.to_numpy(), mutated.to_numpy())

    def test_settlement_uses_actual_prices_without_changing_dispatch(self) -> None:
        timestamps = pd.date_range("2025-12-01", periods=4, freq="15min")
        schedule = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price_eur_mwh": [50.0, 60.0, 100.0, 110.0],
                "p_buy_mw": [1.0, 0.0, 0.0, 0.0],
                "p_sell_mw": [0.0, 0.0, 1.0, 0.0],
                "net_export_mw": [-1.0, 0.0, 1.0, 0.0],
                "buy_energy_mwh": [0.25, 0.0, 0.0, 0.0],
                "sell_energy_mwh": [0.0, 0.0, 0.25, 0.0],
                "soc_mwh": [1.23, 1.23, 0.96, 0.96],
                "soc_pct": [0.615, 0.615, 0.48, 0.48],
                "degradation_cost_eur": [0.0, 0.0, 1.0, 0.0],
                "gross_revenue_eur": [0.0, 0.0, 25.0, 0.0],
                "gross_purchase_eur": [12.5, 0.0, 0.0, 0.0],
                "interval_profit_eur": [-12.5, 0.0, 24.0, 0.0],
                "operation": ["buy", "idle", "sell", "idle"],
            }
        )
        actual_prices = pd.Series([40.0, 60.0, 120.0, 110.0], index=timestamps)

        settled = settle_schedule_on_actual_prices(schedule, actual_prices)

        self.assertEqual(settled["p_buy_mw"].tolist(), schedule["p_buy_mw"].tolist())
        self.assertEqual(settled["p_sell_mw"].tolist(), schedule["p_sell_mw"].tolist())
        self.assertAlmostEqual(settled["actual_gross_purchase_eur"].sum(), 10.0)
        self.assertAlmostEqual(settled["actual_gross_revenue_eur"].sum(), 30.0)
        self.assertAlmostEqual(settled["actual_interval_profit_eur"].sum(), 19.0)

        trades = build_trade_ledger(settled, target_day="2025-12-01")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["sequence"], "buy->sell")
        self.assertEqual(trades.iloc[0]["trade_type"], "completed_buy_sell")
        self.assertTrue(bool(trades.iloc[0]["is_completed_buy_sell"]))
        self.assertEqual(trades.iloc[0]["actual_profit_class"], "profitable")

    def test_end_to_end_backtest_writes_expected_statistics_shape(self) -> None:
        history = build_synthetic_history(days=38, start_date="2025-11-01")
        frame = history.rename(
            columns={
                "timestamp": "DELIVERY_MTU",
                "price_eur_mwh": "DAM_Price_EUR_MWh",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "prices.csv"
            frame.to_csv(input_path, index=False)
            result = run_forecast_strategy_backtest(
                input_file=input_path,
                start_date="2025-12-05",
                end_date="2025-12-07",
                backtest_days=None,
                window_days=20,
                degradation_source="dummy",
                installed_capacity_mw=1.0,
            )

        self.assertEqual(len(result.daily_stats), 3)
        self.assertEqual(len(result.interval_schedule), 3 * 96)
        self.assertEqual(len(result.summary), 1)
        self.assertIn("actual_net_profit_eur", result.summary.columns)
        self.assertIn("profitable_trade_ratio", result.summary.columns)
        self.assertIn("completed_buy_sell_profitable_trade_ratio", result.summary.columns)
        self.assertIn("excluded_non_buy_sell_ledger_rows", result.summary.columns)
        self.assertTrue((result.daily_stats["solver_status"] == "Optimal").all())


if __name__ == "__main__":
    unittest.main()
