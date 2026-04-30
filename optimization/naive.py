# Benchmark 2 - Naive Optimization System

# Threshold with the EMA filter

import pandas as pd
import numpy as np
from config import params

def bess_ema(prices, alpha_factor=0.3,margin=0.05):
    TIME_STEPS = len(prices)
    ema_prices = pd.Series(prices).ewm(alpha=alpha_factor, adjust=False).mean().values
    p_buy = np.zeros(TIME_STEPS)
    p_sell = np.zeros(TIME_STEPS)
    soc = np.zeros(TIME_STEPS)
    current_soc = params['soc_init'] * params['e_max']
    profit = 0.0
    for t in range(TIME_STEPS):
        current_price = prices[t]
        thres = ema_prices[t]
        # Statistical Arbitrage Logic
        if current_price > thres + margin: 
            available_energy = current_soc - (params['soc_min'] * params['e_max'])
            discharge_mw = max(0,min(params['p_max'], available_energy * params['eta_dis'] / params['dt']))
            p_sell[t] = discharge_mw
            current_soc -= (discharge_mw / params['eta_dis']) * params['dt']
            profit += discharge_mw * params['dt'] * current_price
        elif current_price < thres - margin: 
            available_space = (params['soc_max'] * params['e_max']) - current_soc
            charge_mw = max(0,min(params['p_max'], available_space / (params['eta_ch'] * params['dt'])))
            
            p_buy[t] = charge_mw
            current_soc += charge_mw * params['eta_ch'] * params['dt']
            profit -= charge_mw * params['dt'] * current_price
            
        soc[t] = current_soc

    return profit, p_buy, p_sell, soc






