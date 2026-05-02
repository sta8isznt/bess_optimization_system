import pulp as pl

from .config import params


DEFAULT_ENERGY_POINTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # (MWh)
DEFAULT_COST_POINTS = [0.0, 0.10, 0.25, 0.50, 0.90, 1.50]  # (€)


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


def bess_order(
    prices,
    battery_params=None,
    degradation_energy_points=None,
    degradation_cost_points=None,
    solver=None,
    solver_msg=False,
    terminal_soc_mode=None,
):
    prices = [float(price) for price in prices]
    if not prices:
        raise ValueError("prices must contain at least one timestep.")

    TIME_STEPS = range(len(prices)) #96
    bess_params = dict(params if battery_params is None else battery_params)
    if "dt" not in bess_params and "DT" in bess_params:
        bess_params["dt"] = bess_params["DT"]
    terminal_mode = (terminal_soc_mode or bess_params.get("terminal_soc_mode", "equal_initial")).strip().lower()
    if terminal_mode not in {"equal_initial", "free"}:
        raise ValueError('terminal_soc_mode must be "equal_initial" or "free".')

    # look in the LUT  - EXAMPLE:
    energy_points = DEFAULT_ENERGY_POINTS if degradation_energy_points is None else degradation_energy_points
    cost_points = DEFAULT_COST_POINTS if degradation_cost_points is None else degradation_cost_points
    if len(energy_points) != len(cost_points):
        raise ValueError("degradation_energy_points and degradation_cost_points must have the same length.")
    if len(energy_points) < 2:
        raise ValueError("At least two degradation curve points are required.")
    K = range(len(energy_points)) 

    obj_f = pl.LpProblem("BESS_Optimization_Engine", pl.LpMaximize)

    # Variables
    p_buy = pl.LpVariable.dicts("P_buy", TIME_STEPS, lowBound=0, upBound=bess_params['p_max'], cat=pl.LpContinuous)
    p_sell = pl.LpVariable.dicts("P_sell", TIME_STEPS, lowBound=0, upBound=bess_params['p_max'], cat=pl.LpContinuous) 
    soc = pl.LpVariable.dicts("SoC", TIME_STEPS, lowBound=bess_params['soc_min'] * bess_params['e_max'], upBound=bess_params['soc_max'] * bess_params['e_max'], cat=pl.LpContinuous)
    u = pl.LpVariable.dicts("u", TIME_STEPS, cat=pl.LpBinary)   

    lambdas = pl.LpVariable.dicts("Lambda", (TIME_STEPS, K), lowBound=0, upBound=1, cat=pl.LpContinuous)
    dynamic_deg_cost = pl.LpVariable.dicts("Degradation_Cost", TIME_STEPS, lowBound=0, cat=pl.LpContinuous)

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

        # SOS2 CONSTRAINTS 
        obj_f += pl.lpSum([lambdas[t][k] for k in K]) == 1, f"Lambda_Sum_{t}"
        obj_f += (p_sell[t] * bess_params['dt']) == pl.lpSum([lambdas[t][k] * energy_points[k] for k in K]), f"Energy_Interp_{t}"
        obj_f += dynamic_deg_cost[t] == pl.lpSum([lambdas[t][k] * cost_points[k] for k in K]), f"Cost_Interp_{t}"

    # Cycle Continuity Constraint
    if terminal_mode == "equal_initial":
        last_step = TIME_STEPS[-1]
        obj_f += soc[last_step] == bess_params['soc_init'] * bess_params['e_max'], "Cycle_Continuity"

    # Objective Function: Maximize Total Profit
    obj_f += pl.lpSum([
        (prices[t] * (p_sell[t] - p_buy[t]) * bess_params['dt']) - dynamic_deg_cost[t] 
        for t in TIME_STEPS
    ]), "Total_Profit"
    
    obj_f.solve(solver or default_solver(msg=solver_msg))
    
    return obj_f, p_buy, p_sell, soc, dynamic_deg_cost
