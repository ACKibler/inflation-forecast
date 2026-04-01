import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load and prepare stationary series ────────────────────────────────────
df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)

# Fill the Oct-2025 gap via interpolation so index is gapless
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
df = df.reindex(full_idx).interpolate(method="time")
df.index.freq = "MS"

# Stationarity transformations (from ADF analysis)
stationary = pd.DataFrame(index=df.index)
stationary["DCPI"]     = np.log(df["CPI"]).diff()       # log-diff CPI ~ monthly inflation
stationary["DUNRATE"]  = df["UNRATE"].diff()             # first-diff (borderline stationary in levels, cleaner differenced for VAR)
stationary["DM2"]      = np.log(df["M2"]).diff()         # log-diff M2
stationary["OIL"]      = df["OIL"]                       # stationary in levels
stationary["FEDFUNDS"] = df["FEDFUNDS"]                  # stationary in levels
stationary.dropna(inplace=True)

print(f"Stationary dataset: {stationary.shape[0]} obs x {stationary.shape[1]} vars")
print(stationary.describe().round(4))

# ── 2. Train / test split (80/20) ─────────────────────────────────────────────
split = int(len(stationary) * 0.8)
train = stationary.iloc[:split]
test  = stationary.iloc[split:]
print(f"\nTrain: {train.index[0].date()} to {train.index[-1].date()}  ({len(train)} obs)")
print(f"Test : {test.index[0].date()}  to {test.index[-1].date()}   ({len(test)} obs)")

# ── 3. Lag order selection ────────────────────────────────────────────────────
model_select = VAR(train)
lag_results  = model_select.select_order(maxlags=12)
print("\nLag order selection:")
print(lag_results.summary())

best_lag = lag_results.selected_orders["aic"]
print(f"\nSelected lag (AIC): {best_lag}")

# ── 4. Fit VAR model ──────────────────────────────────────────────────────────
var_model = VAR(train)
fitted    = var_model.fit(best_lag)
print(fitted.summary())

# Durbin-Watson on residuals (check for serial correlation)
dw = durbin_watson(fitted.resid)
print("\nDurbin-Watson statistics per variable:")
for var, stat in zip(stationary.columns, dw):
    print(f"  {var:<12} {stat:.4f}  {'ok' if 1.5 < stat < 2.5 else 'WARNING'}")

# ── 5. Impulse Response Functions ─────────────────────────────────────────────
irf = fitted.irf(periods=24)
fig = irf.plot(orth=True, figsize=(14, 12), signif=0.05)
fig.suptitle("Orthogonalised Impulse Response Functions (24 months)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/var_irf.png", bbox_inches="tight")
plt.show()
print("IRF plot saved.")

# ── 6. Out-of-sample forecast ─────────────────────────────────────────────────
n_test   = len(test)
lag_vals = train.values[-best_lag:]          # seed values for recursive forecast
fc_array = fitted.forecast(lag_vals, steps=n_test)
fc_df    = pd.DataFrame(fc_array, index=test.index, columns=stationary.columns)

# ── 7. Back-transform DCPI to CPI levels ──────────────────────────────────────
# log(CPI_t) = log(CPI_{t-1}) + DCPI_t  => exp(cumsum) from last known log-level
last_log_cpi   = np.log(df["CPI"]).iloc[split]      # last log-CPI in train
fc_log_cpi     = last_log_cpi + fc_df["DCPI"].cumsum()
fc_cpi_level   = np.exp(fc_log_cpi)
actual_cpi_level = df["CPI"].iloc[split + 1 : split + 1 + n_test]

# align index lengths (first forecast step corresponds to test row 0)
min_len = min(len(fc_cpi_level), len(actual_cpi_level))
fc_cpi_level     = fc_cpi_level.iloc[:min_len]
actual_cpi_level = actual_cpi_level.iloc[:min_len]

# ── 8. Error metrics ──────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(actual_cpi_level, fc_cpi_level))
mae  = mean_absolute_error(actual_cpi_level, fc_cpi_level)
mape = np.mean(np.abs((actual_cpi_level - fc_cpi_level) / actual_cpi_level)) * 100
print(f"\nVAR Forecast Errors (CPI levels):")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2f}%")

# ── 9. Forecast plot ──────────────────────────────────────────────────────────
train_cpi = df["CPI"].iloc[:split + 1]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train_cpi.index, train_cpi, color="#1f77b4", linewidth=1.2, label="Train")
ax.plot(actual_cpi_level.index, actual_cpi_level, color="#2ca02c",
        linewidth=1.5, label="Actual (Test)")
ax.plot(fc_cpi_level.index, fc_cpi_level, color="#d62728", linewidth=1.5,
        linestyle="--", label="VAR Forecast")

ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title(f"VAR({best_lag}) — CPI Forecast vs Actual\n"
             f"RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%", fontsize=12)
ax.set_ylabel("CPI")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/var_forecast.png", bbox_inches="tight")
plt.show()

# ── 10. All-variable forecast plot ───────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

for ax, col, color in zip(axes, stationary.columns, colors):
    ax.plot(test.index, test[col], color="black", linewidth=1.2, label="Actual")
    ax.plot(fc_df.index, fc_df[col], color=color, linewidth=1.4,
            linestyle="--", label="Forecast")
    ax.set_ylabel(col, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

axes[-1].set_xlabel("Date")
fig.suptitle(f"VAR({best_lag}) — All Variable Forecasts vs Actual", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/var_all_forecasts.png", bbox_inches="tight")
plt.show()

print("\nOutputs saved:")
print("  outputs/var_irf.png")
print("  outputs/var_forecast.png")
print("  outputs/var_all_forecasts.png")
