"""Build the cleaned 15-minute DAM price signal from local HeNEx XLSX files."""

from __future__ import annotations

import pandas as pd

from bess_optimization.io.henex import henexing
from bess_optimization.paths import CLEANED_DATA_DIR, HENEX_DATA_DIR


def main() -> None:
    folder_2024 = HENEX_DATA_DIR / "Results_2024" / "DAM"
    folder_2025 = HENEX_DATA_DIR / "Results_2025" / "DAM"

    df_raw_total = pd.concat([henexing(folder_2024), henexing(folder_2025)], ignore_index=True)
    df_raw_total["DELIVERY_MTU"] = pd.to_datetime(df_raw_total["DELIVERY_MTU"])
    df_clean = df_raw_total.groupby("DELIVERY_MTU").mean().sort_index()
    df_15min = df_clean.resample("15min").ffill()
    df_15min.columns = ["DAM_Price_EUR_MWh"]
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEANED_DATA_DIR / "price_signals_15m.csv"
    df_15min.to_csv(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
