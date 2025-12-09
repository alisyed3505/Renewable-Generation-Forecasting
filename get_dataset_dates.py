import pandas as pd
import os
import glob

def get_dates():
    data_dir = r'c:\Users\haide\Desktop\PV ENERGY FORECASTING\Solar-Power-Forecasting\GermanSolarFarm\data'
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not files:
        print("No CSV files found.")
        return

    # Check the first file to see structure
    first_file = files[0]
    try:
        df = pd.read_csv(first_file)
        print(f"Columns: {df.columns.tolist()}")
        
        # specific column for date?
        # usually 'date', 'timestamp', 'time'
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            print(f"Date column found: {date_col}")
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"Start Date: {df[date_col].min()}")
            print(f"End Date: {df[date_col].max()}")
        else:
            print("No date column identified automatically.")
            print("First 5 rows:")
            print(df.head())
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    get_dates()
