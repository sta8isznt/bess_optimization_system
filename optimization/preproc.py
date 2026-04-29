import pandas as pd
import glob
import os

def henexing(folder_path):
    
    file_pattern = os.path.join(folder_path, "*.xlsx")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"Not found {folder_path}!")
        return pd.DataFrame(columns=['DELIVERY_MTU', 'MCP'])

    df_list = []
    
    for file in all_files:
        try:
            daily_df = pd.read_excel(file)
            clean_daily = daily_df[['DELIVERY_MTU', 'MCP']].drop_duplicates()
            df_list.append(clean_daily)
        except Exception as e:
            print(f"Error at {file}: {e}")
            
    if not df_list:
        return pd.DataFrame(columns=['DELIVERY_MTU', 'MCP'])

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
