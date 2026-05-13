from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from bess_optimization.forecasting.dam_15min_forecast import build_synthetic_history
from bess_optimization.models import BatteryConfig, OptimizationResult


DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from optimizer_adapter import OptimizationRequest, run_optimization  # noqa: E402


def _write_price_file(path: Path, days: int = 3) -> None:
    history = build_synthetic_history(days=days, start_date="2025-11-01")
    history.rename(
        columns={
            "timestamp": "DELIVERY_MTU",
            "price_eur_mwh": "DAM_Price_EUR_MWh",
        }
    ).to_csv(path, index=False)


class DashboardAdapterTests(unittest.TestCase):
    def test_dashboard_adapter_returns_optimization_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            _write_price_file(price_path)

            result = run_optimization(
                OptimizationRequest(
                    run_mode="daily",
                    target_date="2025-11-02",
                    price_file=price_path,
                    degradation_source="dummy",
                    battery=BatteryConfig(dt=0.25),
                    scale_capacity_mw=1.0,
                )
            )

        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(len(result.dispatch_df), 96)
        self.assertIn("park_net_profit_eur", result.summary_dict)


if __name__ == "__main__":
    unittest.main()
