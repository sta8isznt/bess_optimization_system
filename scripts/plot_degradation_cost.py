import pandas as pd
import matplotlib.pyplot as plt

# Update the path to your CSV
path = "data/processed/engine_degradation_points.csv"

df = pd.read_csv(path)

# Pick the column you want to plot
value_col = "deg_cost_eur_per_mwh"
if value_col not in df.columns:
    # fallback candidates
    for col in ["v_deg_eur_per_mwh", "deg_cost_final", "deg_cost_eur_per_MWh_throughput"]:
        if col in df.columns:
            value_col = col
            break

# Plot by temperature if available
if "temperature_c" in df.columns:
    for temp, g in df.groupby("temperature_c"):
        g = g.dropna(subset=["dod", value_col]).sort_values("dod")
        plt.plot(g["dod"], g[value_col], marker="o", label=str(temp))
    plt.legend(title="Temperature C")
else:
    g = df.dropna(subset=["dod", value_col]).sort_values("dod")
    plt.plot(g["dod"], g[value_col], marker="o")

plt.xlabel("DoD")
plt.ylabel("Degradation cost (EUR/MWh)")
plt.title("Degradation cost vs DoD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Pivot table
if "temperature_c" in df.columns:
    table = df.pivot_table(index="dod", columns="temperature_c", values=value_col, aggfunc="mean").sort_index()
else:
    table = df[["dod", value_col]].sort_values("dod").reset_index(drop=True)

print("\nPivot table:")
print(table.to_string())