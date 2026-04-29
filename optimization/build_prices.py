import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bess_optimization_system.optimization.preproc import henexing

BASE_DIR = Path(__file__).resolve().parent
HENEX_DATA_DIR = BASE_DIR / "data" / "HeNex_data"
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned_data"

folder_2024 = HENEX_DATA_DIR / "Results_2024" / "DAM"
folder_2025 = HENEX_DATA_DIR / "Results_2025" / "DAM"

df_24 = henexing(folder_2024)
df_25 = henexing(folder_2025)

df_raw_total = pd.concat([df_24, df_25], ignore_index=True)

df_raw_total['DELIVERY_MTU'] = pd.to_datetime(df_raw_total['DELIVERY_MTU'])

df_clean = df_raw_total.groupby('DELIVERY_MTU').mean()

df_clean = df_clean.sort_index()

df_15min = df_clean.resample('15min').ffill()

df_15min.columns = ['DAM_Price_EUR_MWh'] 
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
df_15min.to_csv(CLEANED_DATA_DIR / 'price_signals_15m.csv')
