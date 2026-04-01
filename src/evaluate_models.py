"""
evaluate_models.py
Runs both ARIMA and VAR forecasts from scratch and produces a
side-by-side comparison — metrics table + overlay plot.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# ── helpers ───────────────────────────────────────────────────────────────────
def metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae  = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
df = df.reindex(full_idx).interpolate(method="time")
df.index.freq = "MS"

SPLIT = int(len(df) * 0.8)

# ── 2. ARIMA forecast ─────────────────────────────────────────────────────────
log_cpi    = np.log(df["CPI"])
train_arima = log_cpi.iloc[:SPLIT]
test_arima  = log_cpi.iloc[SPLIT:]

print("Fitting ARIMA — grid search ...")
grid = []
for p, q in product(range(5), range(5)):
    try:
        m = ARIMA(train_arima, order=(p, 1, q)).fit()
        grid.append((m.aic, p, q))
    except Exception:
        pass
_, best_p, best_q = min(grid)
print(f"  Best: ARIMA({best_p},1,{best_q})")

arima_fit  = ARIMA(train_arima, order=(best_p, 1, best_q)).fit()
fc_arima   = np.exp(arima_fit.get_forecast(steps=len(test_arima)).predicted_mean)
actual_cpi = np.exp(test_arima)
arima_metrics = metrics(actual_cpi, fc_arima)

# ── 3. VAR forecast ───────────────────────────────────────────────────────────
stationary = pd.DataFrame(index=df.index)
stationary["DCPI"]     = np.log(df["CPI"]).diff()
stationary["DUNRATE"]  = df["UNRATE"].diff()
stationary["DM2"]      = np.log(df["M2"]).diff()
stationary["OIL"]      = df["OIL"]
stationary["FEDFUNDS"] = df["FEDFUNDS"]
stationary.dropna(inplace=True)

train_var = stationary.iloc[:SPLIT]
test_var  = stationary.iloc[SPLIT:]

print("Fitting VAR — lag selection ...")
best_lag = VAR(train_var).select_order(maxlags=12).selected_orders["aic"]
print(f"  Best lag: {best_lag}")

var_fit  = VAR(train_var).fit(best_lag)
fc_array = var_fit.forecast(train_var.values[-best_lag:], steps=len(test_var))
fc_df    = pd.DataFrame(fc_array, index=test_var.index, columns=stationary.columns)

last_log_cpi = np.log(df["CPI"]).iloc[SPLIT]
fc_cpi_var   = np.exp(last_log_cpi + fc_df["DCPI"].cumsum())
actual_var   = df["CPI"].iloc[SPLIT + 1 : SPLIT + 1 + len(fc_cpi_var)]

min_len    = min(len(fc_cpi_var), len(actual_var))
fc_cpi_var = fc_cpi_var.iloc[:min_len]
actual_var = actual_var.iloc[:min_len]
var_metrics = metrics(actual_var, fc_cpi_var)

# ── 4. Metrics table ──────────────────────────────────────────────────────────
results = pd.DataFrame(
    {f"ARIMA({best_p},1,{best_q})": arima_metrics,
     f"VAR({best_lag})":            var_metrics}
).T.round(4)

print("\n" + "=" * 50)
print("  MODEL COMPARISON")
print("=" * 50)
print(results.to_string())
print("=" * 50)

winner = results["RMSE"].idxmin()
print(f"\nBest model by RMSE: {winner}")

# ── 5. Comparison plot — both forecasts on one chart ─────────────────────────
train_cpi_level = df["CPI"].iloc[:SPLIT + 1]

fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                         gridspec_kw={"height_ratios": [3, 1]})

# Top: forecast overlay
ax = axes[0]
ax.plot(train_cpi_level.index, train_cpi_level,
        color="#1f77b4", linewidth=1.2, label="Train (actual)")
ax.plot(actual_cpi.index, actual_cpi,
        color="#2ca02c", linewidth=2.0, label="Test (actual)")
ax.plot(fc_arima.index, fc_arima,
        color="#ff7f0e", linewidth=1.5, linestyle="--",
        label=f"ARIMA({best_p},1,{best_q}) forecast")
ax.plot(fc_cpi_var.index, fc_cpi_var,
        color="#d62728", linewidth=1.5, linestyle=":",
        label=f"VAR({best_lag}) forecast")

ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("CPI")
ax.set_title("ARIMA vs VAR — CPI Forecast Comparison", fontsize=13)
ax.legend(fontsize=10)

# Bottom: absolute errors
arima_err = (actual_cpi - fc_arima).abs().reindex(actual_var.index).dropna()
var_err   = (actual_var - fc_cpi_var).abs()
shared    = arima_err.index.intersection(var_err.index)

ax2 = axes[1]
ax2.plot(shared, arima_err.loc[shared],
         color="#ff7f0e", linewidth=1.2, label="ARIMA |error|")
ax2.plot(shared, var_err.loc[shared],
         color="#d62728", linewidth=1.2, linestyle=":", label="VAR |error|")
ax2.set_ylabel("|Error| (CPI pts)")
ax2.set_title("Absolute Forecast Error", fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", bbox_inches="tight")
plt.show()

# ── 6. Bar chart — metric comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
metrics_list = ["RMSE", "MAE", "MAPE (%)"]
colors = ["#ff7f0e", "#d62728"]

for ax, metric in zip(axes, metrics_list):
    vals = results[metric]
    bars = ax.bar(vals.index, vals.values, color=colors, width=0.4, edgecolor="white")
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * val,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, fontsize=9)

fig.suptitle("Model Performance Metrics", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/metrics_comparison.png", bbox_inches="tight")
plt.show()

print("\nOutputs saved:")
print("  outputs/model_comparison.png")
print("  outputs/metrics_comparison.png")

# save metrics to CSV for dashboard use
results.to_csv("outputs/model_metrics.csv")
print("  outputs/model_metrics.csv")
