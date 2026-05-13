from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bess_optimization.models import BatteryConfig, ForecastRequest, OptimizationRequest, OptimizationResult
from bess_optimization.io.degradation import default_lut_for_source, load_degradation_curve
from bess_optimization.io.prices import load_price_signal_day
from bess_optimization.paths import DEFAULT_PYBAMM_LUT_PATH
from bess_optimization.forecasting.dam_15min_forecast import build_synthetic_history
from bess_optimization.services.forecasting import run_forecast
from bess_optimization.services.optimization import run_optimization
from bess_optimization.degradation.pybamm_lut import PyBaMMLutConfig, validate_config


def _write_price_file(path: Path, days: int = 45) -> None:
    history = build_synthetic_history(days=days, start_date="2025-11-01")
    history.rename(
        columns={
            "timestamp": "DELIVERY_MTU",
            "price_eur_mwh": "DAM_Price_EUR_MWh",
        }
    ).to_csv(path, index=False)


class SourceWorkflowTests(unittest.TestCase):
    def test_run_optimization_dispatches_annual_mode(self) -> None:
        expected = OptimizationResult(
            status="Optimal",
            dispatch_df=pd.DataFrame(),
            summary_dict={},
            params_used={},
            files_used={},
        )

        with patch(
            "bess_optimization.services.optimization._run_annual_optimization",
            return_value=expected,
        ) as annual_runner:
            result = run_optimization(OptimizationRequest(run_mode="annual"))

        self.assertIs(result, expected)
        annual_runner.assert_called_once()

    def test_price_loader_builds_complete_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            _write_price_file(price_path, days=3)

            prices = load_price_signal_day(price_path, "2025-11-02", dt=0.25)

        self.assertEqual(len(prices), 96)
        self.assertFalse(prices.isna().any())

    def test_degradation_dummy_curve_matches_optimizer_interval(self) -> None:
        curve = load_degradation_curve(
            source="dummy",
            params={"p_max": 1.0, "e_max": 2.0, "dt": 0.25},
        )

        self.assertGreaterEqual(len(curve.energy_points), 2)
        self.assertIsInstance(curve.energy_points, tuple)
        self.assertIsInstance(curve.cost_points, tuple)
        self.assertEqual(curve.energy_points[0], 0.0)
        self.assertEqual(curve.source_label, "synthetic dummy degradation curve")

    def test_pybamm_degradation_source_uses_default_lut(self) -> None:
        curve = load_degradation_curve(
            source="pybamm",
            params={"p_max": 1.0, "e_max": 2.0, "dt": 0.25},
            temperature_c=25.0,
        )

        self.assertTrue(DEFAULT_PYBAMM_LUT_PATH.exists())
        self.assertEqual(default_lut_for_source("pybamm"), DEFAULT_PYBAMM_LUT_PATH)
        self.assertIsNone(default_lut_for_source("unknown"))
        self.assertGreaterEqual(len(curve.energy_points), 2)
        self.assertEqual(curve.energy_points[0], 0.0)
        self.assertIn(DEFAULT_PYBAMM_LUT_PATH.name, curve.source_label)

    def test_forecast_service_writes_optimizer_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            output_path = Path(tmp) / "forecast.csv"
            _write_price_file(price_path, days=40)

            result = run_forecast(
                ForecastRequest(
                    input_file=price_path,
                    output_file=output_path,
                    target_date="2025-12-10",
                    window_days=20,
                )
            )
            output_exists = result.output_path.exists()
            optimizer_input_exists = result.optimizer_input_path.exists()

        self.assertEqual(len(result.forecast), 96)
        self.assertIsNotNone(result.output_path)
        self.assertIsNotNone(result.optimizer_input_path)
        self.assertTrue(output_exists)
        self.assertTrue(optimizer_input_exists)

    def test_daily_optimization_service_runs_on_synthetic_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            _write_price_file(price_path, days=3)

            result = run_optimization(
                OptimizationRequest(
                    run_mode="daily",
                    target_date="2025-11-02",
                    price_file=price_path,
                    degradation_source="dummy",
                    battery=BatteryConfig(
                        p_max=1.0,
                        e_max=2.0,
                        eta_ch=0.92,
                        eta_dis=0.92,
                        soc_min=0.1,
                        soc_max=0.9,
                        soc_init=0.5,
                        dt=0.25,
                    ),
                    scale_capacity_mw=1.0,
                )
            )

        self.assertEqual(result.status, "Optimal")
        self.assertEqual(len(result.dispatch_df), 96)
        self.assertIn("park_net_profit_eur", result.summary_dict)
        self.assertIsNotNone(result.benchmark_comparison_df)

    def test_forecast_then_optimize_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            forecast_path = Path(tmp) / "forecast.csv"
            _write_price_file(price_path, days=40)
            forecast_result = run_forecast(
                ForecastRequest(
                    input_file=price_path,
                    output_file=forecast_path,
                    target_date="2025-12-10",
                    window_days=20,
                )
            )
            optimization_result = run_optimization(
                OptimizationRequest(
                    run_mode="daily",
                    target_date="2025-12-10",
                    price_file=forecast_result.optimizer_input_path,
                    degradation_source="dummy",
                    battery=BatteryConfig(dt=0.25),
                    scale_capacity_mw=1.0,
                )
            )

        self.assertEqual(optimization_result.status, "Optimal")
        self.assertEqual(len(optimization_result.dispatch_df), 96)

    def test_pybamm_lut_config_validation_without_simulation(self) -> None:
        config = PyBaMMLutConfig()
        validate_config(config)

        invalid = PyBaMMLutConfig(dod_breakpoints=(0.50,))
        with self.assertRaises(ValueError):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
