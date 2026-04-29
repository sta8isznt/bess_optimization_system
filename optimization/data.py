import pandas as pd
from bess_optimization_system.optimization.preproc import henexing

folder_2024 = "Results_2024/DAM" 
folder_2025 = "Results_2025/DAM" 

df_24 = henexing(folder_2024)
df_25 = henexing(folder_2025)

df_raw_total = pd.concat([df_24, df_25], ignore_index=True)

df_raw_total['DELIVERY_MTU'] = pd.to_datetime(df_raw_total['DELIVERY_MTU'])

df_clean = df_raw_total.groupby('DELIVERY_MTU').mean()

df_clean = df_clean.sort_index()

df_15min = df_clean.resample('15min').ffill()

df_15min.columns = ['DAM_Price_EUR_MWh'] 
df_15min.to_csv('price_signals_15m.csv')
