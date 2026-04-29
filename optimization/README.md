# OPTIMIZATION 

## PHASE 1:  Data for Backtesting our optimized BESS Strategy 

* We downloaded data from EnEx for 2024-2025:

> * Historical Scope: We aggregated and unified DAM data covering the entire year of 2024 and the first quarter of 2025. This extensive dataset ensures that the algorithm is rigorously tested against full seasonal cycles, capturing the dynamics of both high-demand summer peaks and varied winter loads.

> * Time Alignment: Hourly price signals were transformed into **15-minute** Market Time Units (MTU) using the Forward Fill technique. This step was crucial to align our inputs with the modern 15-minute settlement requirements of the Independent Power Transmission Operator.


