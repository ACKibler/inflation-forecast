import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
# Reindex to a complete monthly range so the freq is unambiguous
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
df = df.reindex(full_idx).interpolate(method="time")
df.index.freq = "MS"

log_cpi = np.log(df["CPI"])          # log-level fed into ARIMA(p,1,q)
cpi_raw = df["CPI"]

# Train / test split (80/20)
split = int(len(log_cpi) * 0.8)
train = log_cpi.iloc[:split]
test  = log_cpi.iloc[split:]
print(f"Train: {train.index[0].date()} to {train.index[-1].date()}  ({len(train)} obs)")
print(f"Test : {test.index[0].date()}  to {test.index[-1].date()}   ({len(test)} obs)")

# ── 2. AIC/BIC grid search over p, d=1, q ────────────────────────────────────
print("\nGrid searching ARIMA(p,1,q)  p,q in 0..4 ...")
results = []
for p, q in product(range(5), range(5)):
    try:
        m = ARIMA(train, order=(p, 1, q)).fit()
        results.append({"p": p, "q": q, "AIC": m.aic, "BIC": m.bic})
    except Exception:
        pass

grid = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
print("\nTop 10 by AIC:")
print(grid.head(10).to_string(index=False))

best = grid.iloc[0]
best_p, best_q = int(best.p), int(best.q)
print(f"\nBest order: ARIMA({best_p}, 1, {best_q})  AIC={best.AIC:.2f}  BIC={best.BIC:.2f}")

# ── 3. Fit final model on full train set ──────────────────────────────────────
model = ARIMA(train, order=(best_p, 1, best_q)).fit()
print("\n" + model.summary().tables[1].as_text())

# ── 4. Residual diagnostics ───────────────────────────────────────────────────
residuals = model.resid

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Residuals over time
axes[0, 0].plot(residuals.index, residuals, linewidth=0.9, color="#1f77b4")
axes[0, 0].axhline(0, color="black", linewidth=0.7, linestyle="--")
axes[0, 0].set_title("Residuals over Time")

# Histogram
axes[0, 1].hist(residuals, bins=35, color="#1f77b4", edgecolor="white")
axes[0, 1].set_title("Residual Distribution")

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(residuals, lags=24, ax=axes[1, 0], title="ACF of Residuals")
plot_pacf(residuals, lags=24, ax=axes[1, 1], title="PACF of Residuals", method="ywm")

fig.suptitle(f"ARIMA({best_p},1,{best_q}) — Residual Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/arima_diagnostics.png", bbox_inches="tight")
plt.show()

# Ljung-Box test
lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\nLjung-Box test (H0: no autocorrelation in residuals):")
print(lb.to_string())

# ── 5. Out-of-sample forecast ─────────────────────────────────────────────────
n_test = len(test)
forecast_result = model.get_forecast(steps=n_test)
fc_log    = forecast_result.predicted_mean
fc_ci_log = forecast_result.conf_int(alpha=0.05)

# Back-transform to CPI levels
fc_level    = np.exp(fc_log)
fc_ci_lower = np.exp(fc_ci_log.iloc[:, 0])
fc_ci_upper = np.exp(fc_ci_log.iloc[:, 1])
actual_level = np.exp(test)

# ── 6. Error metrics ──────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(actual_level, fc_level))
mae  = mean_absolute_error(actual_level, fc_level)
mape = np.mean(np.abs((actual_level - fc_level) / actual_level)) * 100
print(f"\nForecast Errors (CPI levels):")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2f}%")

# ── 7. Forecast plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

train_level = np.exp(train)
ax.plot(train_level.index, train_level, color="#1f77b4", linewidth=1.2, label="Train")
ax.plot(actual_level.index, actual_level, color="#2ca02c", linewidth=1.5, label="Actual (Test)")
ax.plot(fc_level.index, fc_level, color="#d62728", linewidth=1.5, linestyle="--", label="Forecast")
ax.fill_between(fc_level.index, fc_ci_lower, fc_ci_upper,
                color="#d62728", alpha=0.15, label="95% CI")

ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title(f"ARIMA({best_p},1,{best_q}) — CPI Forecast vs Actual\n"
             f"RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%", fontsize=12)
ax.set_ylabel("CPI")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/arima_forecast.png", bbox_inches="tight")
plt.show()

# ── 8. Save grid search results ───────────────────────────────────────────────
grid.to_csv("outputs/arima_grid_search.csv", index=False)
print("\nOutputs saved:")
print("  outputs/arima_diagnostics.png")
print("  outputs/arima_forecast.png")
print("  outputs/arima_grid_search.csv")
