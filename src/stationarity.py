import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)


def adf_test(series):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "ADF Statistic": round(result[0], 4),
        "p-value":       round(result[1], 4),
        "Lags Used":     result[2],
        "Stationary":    "YES" if result[1] < 0.05 else "NO",
    }


def run_adf(data, label):
    print(f"\n{'='*55}")
    print(f"  ADF Results — {label}")
    print(f"{'='*55}")
    rows = {}
    for col in data.columns:
        r = adf_test(data[col])
        rows[col] = r
        flag = "" if r["Stationary"] == "YES" else "  <-- non-stationary"
        print(f"  {col:<10}  p={r['p-value']:.4f}  {r['Stationary']}{flag}")
    return pd.DataFrame(rows).T


# --- Level ADF ---
results_level = run_adf(df, "Levels")

# --- First differences ---
df_diff = df.diff().dropna()
results_diff = run_adf(df_diff, "First Differences")

# --- Log transform CPI and M2, then difference ---
df_log = df.copy()
df_log[["CPI", "M2"]] = np.log(df[["CPI", "M2"]])
df_log_diff = df_log.diff().dropna()
results_log_diff = run_adf(df_log_diff, "Log-Differenced (CPI, M2 only)")

# --- Plot: levels vs first differences ---
fig, axes = plt.subplots(5, 2, figsize=(14, 14))
colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

for i, (col, color) in enumerate(zip(df.columns, colors)):
    axes[i, 0].plot(df.index, df[col], color=color, linewidth=1.2)
    axes[i, 0].set_title(f"{col} — Level", fontsize=10)

    axes[i, 1].plot(df_diff.index, df_diff[col], color=color, linewidth=1.2, alpha=0.8)
    axes[i, 1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[i, 1].set_title(f"{col} — First Difference", fontsize=10)

plt.suptitle("Levels vs First Differences", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/stationarity.png", bbox_inches="tight", dpi=150)
plt.show()
print("\nPlot saved to outputs/stationarity.png")
