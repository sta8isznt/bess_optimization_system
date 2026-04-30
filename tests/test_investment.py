from __future__ import annotations

import pandas as pd

from investment.investment_model import compute_investment_metrics, compute_npv, compute_payback, load_financial_inputs
from investment.scenarios import apply_scenario


def test_npv_positive():
    cashflows = [-100.0, 60.0, 60.0]
    assert compute_npv(cashflows, 0.05) > 0


def test_payback():
    assert compute_payback(1000.0, 250.0) == 4.0


def test_scenario_scaling():
    metrics = {"annual_profit": 100.0, "degradation_cost": 50.0}
    optimistic = apply_scenario(metrics, "optimistic")
    conservative = apply_scenario(metrics, "conservative")
    assert optimistic["annual_profit"] == 120.0
    assert optimistic["degradation_cost"] == 40.0
    assert conservative["annual_profit"] == 80.0
    assert conservative["degradation_cost"] == 65.0


def test_compute_investment_metrics_shape():
    inputs = load_financial_inputs()
    df = pd.DataFrame([{"net_profit_eur": 1000.0}])
    metrics = compute_investment_metrics(df, inputs)
    assert {"annual_profit", "capex", "opex", "payback", "npv", "irr"} <= set(metrics)
