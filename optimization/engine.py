import pulp as pl
from config import params

def bess_order(prices):
    TIME_STEPS = range(len(prices)) #96

    # look in the LUT  - EXAMPLE:
    energy_points = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # (MWh)
    cost_points   = [0.0, 0.10, 0.25, 0.50, 0.90, 1.50]  # (€)
    K = range(len(energy_points)) 

    obj_f = pl.LpProblem("BESS_Optimization_Engine", pl.LpMaximize)

    # Variables
    p_buy = pl.LpVariable.dicts("P_buy", TIME_STEPS, lowBound=0, upBound=params['p_max'], cat=pl.LpContinuous)
    p_sell = pl.LpVariable.dicts("P_sell", TIME_STEPS, lowBound=0, upBound=params['p_max'], cat=pl.LpContinuous) 
    soc = pl.LpVariable.dicts("SoC", TIME_STEPS, lowBound=params['soc_min'] * params['e_max'], upBound=params['soc_max'] * params['e_max'], cat=pl.LpContinuous)
    u = pl.LpVariable.dicts("u", TIME_STEPS, cat=pl.LpBinary)   

    lambdas = pl.LpVariable.dicts("Lambda", (TIME_STEPS, K), lowBound=0, upBound=1, cat=pl.LpContinuous)
    dynamic_deg_cost = pl.LpVariable.dicts("Degradation_Cost", TIME_STEPS, lowBound=0, cat=pl.LpContinuous)

    # Constraints  
    for t in TIME_STEPS:    
        # No Simultaneous Operation Exclusion
        obj_f += p_buy[t] <= params['p_max'] * u[t], f"Max_Charge_{t}" 
        obj_f += p_sell[t] <= params['p_max'] * (1 - u[t]), f"Max_Discharge_{t}" 
        
        # Energy Balance Constraint
        if t == 0:
            prev_soc = params['soc_init'] * params['e_max'] # MWh
        else:
            prev_soc = soc[t-1]
            
        obj_f += soc[t] == prev_soc + (p_buy[t] * params['eta_ch'] - p_sell[t] / params['eta_dis']) * params['dt'], f"SoC_Balance_{t}"

        # SOS2 CONSTRAINTS 
        obj_f += pl.lpSum([lambdas[t][k] for k in K]) == 1, f"Lambda_Sum_{t}"
        obj_f += (p_sell[t] * params['dt']) == pl.lpSum([lambdas[t][k] * energy_points[k] for k in K]), f"Energy_Interp_{t}"
        obj_f += dynamic_deg_cost[t] == pl.lpSum([lambdas[t][k] * cost_points[k] for k in K]), f"Cost_Interp_{t}"

    # Cycle Continuity Constraint
    last_step = TIME_STEPS[-1]
    obj_f += soc[last_step] == params['soc_init'] * params['e_max'], "Cycle_Continuity"

    # Objective Function: Maximize Total Profit
    obj_f += pl.lpSum([
        (prices[t] * (p_sell[t] - p_buy[t]) * params['dt']) - dynamic_deg_cost[t] 
        for t in TIME_STEPS
    ]), "Total_Profit"
    
    solver = pl.COIN_CMD(path="/opt/homebrew/bin/cbc", msg=False)
    obj_f.solve(solver)
    
    return obj_f, p_buy, p_sell, soc, dynamic_deg_cost