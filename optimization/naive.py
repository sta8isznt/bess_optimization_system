"""EMA threshold heuristic benchmark with no degradation cost."""

import pandas as pd
import numpy as np

try:
    from .config import params
except ImportError:  # Keeps direct execution from the optimization/ folder working.
    from config import params


def _battery_params(battery_params=None):
    bess_params = dict(params if battery_params is None else battery_params)
    if "dt" not in bess_params and "DT" in bess_params:
        bess_params["dt"] = bess_params["DT"]
    return bess_params


def bess_ema(
    prices,
    alpha_factor=0.3,
    margin=0.05,
    battery_params=None,
    initial_soc_mwh=None,
):
    TIME_STEPS = len(prices)
    ema_prices = pd.Series(prices).ewm(alpha=alpha_factor, adjust=False).mean().values
    p_buy = np.zeros(TIME_STEPS)
    p_sell = np.zeros(TIME_STEPS)
    soc = np.zeros(TIME_STEPS)
    bess_params = _battery_params(battery_params)
    if initial_soc_mwh is None:
        current_soc = bess_params["soc_init"] * bess_params["e_max"]
    else:
        current_soc = float(initial_soc_mwh)

    profit = 0.0
    for t in range(TIME_STEPS):
        current_price = prices[t]
        thres = ema_prices[t]
        # Statistical Arbitrage Logic
        if current_price > thres + margin:
            available_energy = current_soc - (
                bess_params["soc_min"] * bess_params["e_max"]
            )
            discharge_mw = max(
                0,
                min(
                    bess_params["p_max"],
                    available_energy * bess_params["eta_dis"] / bess_params["dt"],
                ),
            )
            p_sell[t] = discharge_mw
            current_soc -= (
                discharge_mw / bess_params["eta_dis"]
            ) * bess_params["dt"]
            profit += discharge_mw * bess_params["dt"] * current_price
        elif current_price < thres - margin:
            available_space = (
                bess_params["soc_max"] * bess_params["e_max"]
            ) - current_soc
            charge_mw = max(
                0,
                min(
                    bess_params["p_max"],
                    available_space / (bess_params["eta_ch"] * bess_params["dt"]),
                ),
            )

            p_buy[t] = charge_mw
            current_soc += charge_mw * bess_params["eta_ch"] * bess_params["dt"]
            profit -= charge_mw * bess_params["dt"] * current_price

        soc[t] = current_soc

    return profit, p_buy, p_sell, soc



