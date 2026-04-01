import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# Define series to pull
series = {
    "CPI": "CPIAUCSL",           # Consumer Price Index
    "UNRATE": "UNRATE",          # Unemployment Rate
    "M2": "M2SL",                # M2 Money Supply
    "OIL": "DCOILWTICO",         # WTI Crude Oil Price
    "FEDFUNDS": "FEDFUNDS"       # Federal Funds Rate
}

# Pull data from FRED (2000 onwards)
dfs = []
for name, series_id in series.items():
    data = fred.get_series(series_id, observation_start="2000-01-01")
    data.name = name
    dfs.append(data)

# Combine into one DataFrame
df = pd.concat(dfs, axis=1)
df = df.resample("MS").mean()   # Resample everything to monthly
df.dropna(subset=["CPI", "UNRATE"], inplace=True)

print(df.head())
print(f"\nShape: {df.shape}")

# Save to CSV
df.to_csv("data/macro_data.csv")
print("\nData saved to data/macro_data.csv")