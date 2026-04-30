from __future__ import annotations


SCENARIOS = {
    "base": {"profit_multiplier": 1.0, "degradation_multiplier": 1.0},
    "optimistic": {"profit_multiplier": 1.2, "degradation_multiplier": 0.8},
    "conservative": {"profit_multiplier": 0.8, "degradation_multiplier": 1.3},
}


def apply_scenario(metrics: dict, scenario_name: str) -> dict:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]
    adjusted = dict(metrics)
    profit_multiplier = float(scenario["profit_multiplier"])
    degradation_multiplier = float(scenario["degradation_multiplier"])

    if "annual_profit" in adjusted:
        adjusted["annual_profit"] = float(adjusted["annual_profit"]) * profit_multiplier
    if "net_profit" in adjusted:
        adjusted["net_profit"] = float(adjusted["net_profit"]) * profit_multiplier
    if "degradation_cost" in adjusted:
        adjusted["degradation_cost"] = float(adjusted["degradation_cost"]) * degradation_multiplier
    adjusted["scenario"] = scenario_name
    return adjusted
