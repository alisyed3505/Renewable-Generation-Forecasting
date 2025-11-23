import pandas as pd

try:
    df = pd.read_csv('pv_01.csv', delimiter=';')
    print("Columns:", list(df.columns))
    print("\nShape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nData Types:\n", df.dtypes)
    print("\nTarget Variable (last column):", df.columns[-1])
except Exception as e:
    print(f"Error reading CSV: {e}")
