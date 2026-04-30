# benchmark 1 for our base optimization system

# No degradation optimization

import pulp as pl

from config import params


def default_solver(msg=False):
    homebrew_solver = pl.COIN_CMD(path="/opt/homebrew/bin/cbc", msg=msg)
    if homebrew_solver.available():
        return homebrew_solver

    bundled_solver = pl.PULP_CBC_CMD(msg=msg)
    if bundled_solver.available():
        return bundled_solver

    available_solvers = pl.listSolvers(onlyAvailable=True)
    if available_solvers:
        return pl.getSolver(available_solvers[0], msg=msg)

    raise RuntimeError("No MILP solver is available.")


def bess_order_perfect(
    prices,
    battery_params=None,
    solver=None,
    solver_msg=False,
):
    prices = [float(price) for price in prices]
    if not prices:
        raise ValueError("prices must contain at least one timestep.")

    TIME_STEPS = range(len(prices)) #96
    bess_params = dict(params if battery_params is None else battery_params)
    if "dt" not in bess_params and "DT" in bess_params:
        bess_params["dt"] = bess_params["DT"]

    obj_f = pl.LpProblem("BESS_Optimization_Engine", pl.LpMaximize)

    # Variables
    p_buy = pl.LpVariable.dicts("P_buy", TIME_STEPS, lowBound=0, upBound=bess_params['p_max'], cat=pl.LpContinuous)
    p_sell = pl.LpVariable.dicts("P_sell", TIME_STEPS, lowBound=0, upBound=bess_params['p_max'], cat=pl.LpContinuous) 
    soc = pl.LpVariable.dicts("SoC", TIME_STEPS, lowBound=bess_params['soc_min'] * bess_params['e_max'], upBound=bess_params['soc_max'] * bess_params['e_max'], cat=pl.LpContinuous)
    u = pl.LpVariable.dicts("u", TIME_STEPS, cat=pl.LpBinary)   


    # Constraints  
    for t in TIME_STEPS:    
        # No Simultaneous Operation Exclusion
        obj_f += p_buy[t] <= bess_params['p_max'] * u[t], f"Max_Charge_{t}" 
        obj_f += p_sell[t] <= bess_params['p_max'] * (1 - u[t]), f"Max_Discharge_{t}" 
        
        # Energy Balance Constraint
        if t == 0:
            prev_soc = bess_params['soc_init'] * bess_params['e_max'] # MWh
        else:
            prev_soc = soc[t-1]
            
        obj_f += soc[t] == prev_soc + (p_buy[t] * bess_params['eta_ch'] - p_sell[t] / bess_params['eta_dis']) * bess_params['dt'], f"SoC_Balance_{t}"


    # Cycle Continuity Constraint
    last_step = TIME_STEPS[-1]
    obj_f += soc[last_step] == bess_params['soc_init'] * bess_params['e_max'], "Cycle_Continuity"

    # Objective Function: Maximize Total Profit
    obj_f += pl.lpSum([
        (prices[t] * (p_sell[t] - p_buy[t]) * bess_params['dt'])
        for t in TIME_STEPS
    ]), "Total_Profit"
    
    obj_f.solve(solver or default_solver(msg=solver_msg))
    
    return obj_f, p_buy, p_sell, soc
