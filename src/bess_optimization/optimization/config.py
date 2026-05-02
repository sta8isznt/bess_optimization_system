"""Battery parameters used by the MILP optimization backtests."""

params = {
    "p_max": 1.0,         # Max Power in MW
    "e_max": 2.0,         # Capacity in MWh
    "eta_ch": 0.92,       # Charging Efficiency
    "eta_dis": 0.92,      # Discharging Efficiency
    "soc_min": 0.1,       # 10%
    "soc_max": 0.9,       # 90%
    "soc_init": 0.5,      # Initial SoC (50%)
    "dt": 0.25,           # 15min interval = 0.25 hours
}
