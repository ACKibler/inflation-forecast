"""
paper_analysis.py
Runs all analysis needed to verify and extend paper/paper.md:
  1. Verify ARIMA and VAR numbers
  2. Naive random walk benchmark
  3. Granger causality tests
  4. CUSUM stability test on ARIMA residuals
  5. All figures saved to /figures
"""
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def calc_metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae  = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return rmse, mae, mape

def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
df = df.reindex(full_idx).interpolate(method="time")
df.index.freq = "MS"

SPLIT = int(len(df) * 0.8)
log_cpi = np.log(df["CPI"])

# ── Figure 1: All 5 time series ───────────────────────────────────────────────
labels = {"CPI": "Consumer Price Index", "UNRATE": "Unemployment Rate (%)",
          "M2": "M2 Money Supply (B$)", "OIL": "WTI Crude Oil ($/bbl)",
          "FEDFUNDS": "Federal Funds Rate (%)"}
colors = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd"]

fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
for ax, col, color in zip(axes, df.columns, colors):
    ax.plot(df.index, df[col], color=color, linewidth=1.4)
    ax.set_ylabel(labels[col], fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].set_xlabel("Date")
fig.suptitle("Figure 1: Macroeconomic Series (2000 – Present)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig1_time_series.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig1_time_series.png")

# ── Figure 2: ACF/PACF of log-differenced CPI ────────────────────────────────
dcpi = log_cpi.diff().dropna()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(dcpi, lags=36, ax=ax1, title="Figure 2a: ACF — Log-Differenced CPI")
plot_pacf(dcpi, lags=36, ax=ax2, title="Figure 2b: PACF — Log-Differenced CPI", method="ywm")
plt.tight_layout()
plt.savefig("figures/fig2_acf_pacf_dcpi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig2_acf_pacf_dcpi.png")

# ── 2. ARIMA ─────────────────────────────────────────────────────────────────
sep("ARIMA — Grid Search & Fit")
train_arima = log_cpi.iloc[:SPLIT]
test_arima  = log_cpi.iloc[SPLIT:]

grid = []
for p, q in product(range(5), range(5)):
    try:
        m = ARIMA(train_arima, order=(p, 1, q)).fit()
        grid.append((m.aic, m.bic, p, q))
    except Exception:
        pass
grid.sort()
best_aic, best_bic, best_p, best_q = grid[0]
print(f"Best ARIMA: ({best_p},1,{best_q})  AIC={best_aic:.4f}  BIC={best_bic:.4f}")

arima_fit = ARIMA(train_arima, order=(best_p, 1, best_q)).fit()
fc_arima_log = arima_fit.get_forecast(steps=len(test_arima))
fc_arima     = np.exp(fc_arima_log.predicted_mean)
fc_arima_lo  = np.exp(fc_arima_log.conf_int(alpha=0.05).iloc[:, 0])
fc_arima_hi  = np.exp(fc_arima_log.conf_int(alpha=0.05).iloc[:, 1])
actual_arima = np.exp(test_arima)

arima_rmse, arima_mae, arima_mape = calc_metrics(actual_arima, fc_arima)
print(f"ARIMA RMSE={arima_rmse:.4f}  MAE={arima_mae:.4f}  MAPE={arima_mape:.4f}%")

# CUSUM test on ARIMA residuals
sep("CUSUM Stability Test (ARIMA residuals)")
resid = arima_fit.resid
cusum = resid.cumsum()
n     = len(resid)
sigma = resid.std()
# 5% critical bounds: ±0.948 * sigma * sqrt(n)
bound = 0.948 * sigma * np.sqrt(n)
cusum_breach = (cusum.abs() > bound).any()
print(f"Residual std: {sigma:.6f}")
print(f"CUSUM critical bound (5%): +/- {bound:.6f}")
print(f"Max |CUSUM|: {cusum.abs().max():.6f}")
print(f"Boundary breached: {cusum_breach}  => Parameter {'IN' if cusum_breach else ''}STABILITY detected")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(resid.index, cusum.values, color="#1f77b4", linewidth=1.4, label="CUSUM")
ax.axhline( bound, color="#d62728", linestyle="--", linewidth=1, label="5% critical bounds")
ax.axhline(-bound, color="#d62728", linestyle="--", linewidth=1)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title(f"Figure 5: CUSUM Test — ARIMA({best_p},1,{best_q}) Residuals", fontsize=12)
ax.set_ylabel("Cumulative Sum of Residuals")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig5_cusum.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig5_cusum.png")

# Ljung-Box
lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
print("\nLjung-Box:")
print(lb.to_string())

# ── 3. VAR ────────────────────────────────────────────────────────────────────
sep("VAR — Lag Selection & Fit")
stat = pd.DataFrame(index=df.index)
stat["DCPI"]     = np.log(df["CPI"]).diff()
stat["DUNRATE"]  = df["UNRATE"].diff()
stat["DM2"]      = np.log(df["M2"]).diff()
stat["OIL"]      = df["OIL"]
stat["FEDFUNDS"] = df["FEDFUNDS"]
stat.dropna(inplace=True)

train_var = stat.iloc[:SPLIT]
test_var  = stat.iloc[SPLIT:]

lo = VAR(train_var).select_order(maxlags=12)
best_lag = lo.selected_orders["aic"]
print(f"VAR lag (AIC): {best_lag}")
print(f"VAR lag (BIC): {lo.selected_orders['bic']}")

var_fit  = VAR(train_var).fit(best_lag)
fc_arr   = var_fit.forecast(train_var.values[-best_lag:], steps=len(test_var))
fc_df    = pd.DataFrame(fc_arr, index=test_var.index, columns=stat.columns)

last_log = np.log(df["CPI"]).iloc[SPLIT]
fc_cpi_var = np.exp(last_log + fc_df["DCPI"].cumsum())
actual_var = df["CPI"].iloc[SPLIT + 1: SPLIT + 1 + len(fc_cpi_var)]
min_len    = min(len(fc_cpi_var), len(actual_var))
fc_cpi_var = fc_cpi_var.iloc[:min_len]
actual_var = actual_var.iloc[:min_len]

var_rmse, var_mae, var_mape = calc_metrics(actual_var, fc_cpi_var)
print(f"VAR RMSE={var_rmse:.4f}  MAE={var_mae:.4f}  MAPE={var_mape:.4f}%")

dw = durbin_watson(var_fit.resid)
print("\nDurbin-Watson:")
for v, s in zip(stat.columns, dw): print(f"  {v:<12} {s:.4f}")

# ── 4. Naive random walk benchmark ───────────────────────────────────────────
sep("Naive Random Walk Benchmark")
# Random walk: forecast = last known CPI level (carried forward)
last_train_cpi = df["CPI"].iloc[SPLIT]
rw_forecast    = pd.Series(last_train_cpi, index=actual_var.index)
rw_rmse, rw_mae, rw_mape = calc_metrics(actual_var, rw_forecast)
print(f"Random Walk RMSE={rw_rmse:.4f}  MAE={rw_mae:.4f}  MAPE={rw_mape:.4f}%")

# ── 5. Granger causality tests ────────────────────────────────────────────────
sep("Granger Causality Tests (H0: X does NOT Granger-cause DCPI)")
granger_results = {}
for var in ["DUNRATE", "DM2", "OIL", "FEDFUNDS"]:
    res = var_fit.test_causality("DCPI", causing=var, kind="f")
    granger_results[var] = {
        "F-stat": round(res.test_statistic, 4),
        "p-value": round(res.pvalue, 4),
        "Reject H0 (5%)": "YES" if res.pvalue < 0.05 else "NO"
    }
granger_df = pd.DataFrame(granger_results).T
print(granger_df.to_string())

# ── 6. Summary table ─────────────────────────────────────────────────────────
sep("FINAL MODEL COMPARISON TABLE")
metrics_df = pd.DataFrame({
    "Model":    [f"Random Walk", f"ARIMA({best_p},1,{best_q})", f"VAR({best_lag})"],
    "RMSE":     [round(rw_rmse, 4), round(arima_rmse, 4), round(var_rmse, 4)],
    "MAE":      [round(rw_mae, 4),  round(arima_mae, 4),  round(var_mae, 4)],
    "MAPE (%)": [round(rw_mape, 2), round(arima_mape, 2), round(var_mape, 2)],
})
print(metrics_df.to_string(index=False))

# ── Figure 3: Forecast vs actual (both models + RW) ──────────────────────────
train_cpi_level = df["CPI"].iloc[:SPLIT + 1]

fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(train_cpi_level.index, train_cpi_level,
        color="#1f77b4", linewidth=1.2, label="Train (actual)")
ax.plot(actual_arima.index, actual_arima,
        color="#2ca02c", linewidth=2.0, label="Test (actual)")
ax.plot(fc_arima.index, fc_arima, color="#ff7f0e", linewidth=1.6,
        linestyle="--", label=f"ARIMA({best_p},1,{best_q})")
ax.fill_between(fc_arima.index, fc_arima_lo, fc_arima_hi,
                color="#ff7f0e", alpha=0.12)
ax.plot(fc_cpi_var.index, fc_cpi_var, color="#d62728", linewidth=1.6,
        linestyle=":", label=f"VAR({best_lag})")
ax.plot(rw_forecast.index, rw_forecast, color="#9467bd", linewidth=1.2,
        linestyle="-.", label="Random Walk")
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("CPI")
ax.set_title("Figure 3: CPI Forecast vs Actual — All Models", fontsize=12)
ax.legend(fontsize=9)

ax2 = axes[1]
shared = actual_var.index
arima_err = (actual_arima - fc_arima).abs().reindex(shared).dropna()
var_err   = (actual_var - fc_cpi_var).abs()
rw_err    = (actual_var - rw_forecast).abs()
ax2.plot(arima_err.index, arima_err, color="#ff7f0e", linewidth=1.2,
         label=f"ARIMA |error|")
ax2.plot(var_err.index,   var_err,   color="#d62728", linewidth=1.2,
         linestyle=":", label=f"VAR |error|")
ax2.plot(rw_err.index,    rw_err,    color="#9467bd", linewidth=1.0,
         linestyle="-.", label="RW |error|")
ax2.set_ylabel("|Error| (CPI pts)")
ax2.set_title("Figure 4: Absolute Forecast Error Over Time", fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("figures/fig3_fig4_forecast_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig3_fig4_forecast_error.png")

# ── Figure 6: Selected IRFs (OIL->DCPI, FEDFUNDS->DUNRATE) ───────────────────
irf = var_fit.irf(periods=24)
periods = range(25)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# OIL -> DCPI
irfs_oil_dcpi   = irf.orth_irfs[:, 0, 3]   # response=DCPI(0), impulse=OIL(3)
lower_oil_dcpi  = irf.cum_effect_lower[:, 0, 3] if hasattr(irf, 'cum_effect_lower') else None

ax = axes[0]
ax.plot(periods, irfs_oil_dcpi, color="#ff7f0e", linewidth=2)
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_title("Figure 6a: OIL Shock → DCPI Response", fontsize=11)
ax.set_xlabel("Months after shock")
ax.set_ylabel("Response of DCPI")

# FEDFUNDS -> DUNRATE
irfs_ff_unrate = irf.orth_irfs[:, 1, 4]   # response=DUNRATE(1), impulse=FEDFUNDS(4)
ax2 = axes[1]
ax2.plot(periods, irfs_ff_unrate, color="#9467bd", linewidth=2)
ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax2.set_title("Figure 6b: FEDFUNDS Shock → DUNRATE Response", fontsize=11)
ax2.set_xlabel("Months after shock")
ax2.set_ylabel("Response of DUNRATE")

plt.suptitle("Figure 6: Selected Impulse Response Functions (Orthogonalised)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig6_irf_selected.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig6_irf_selected.png")

# ── Print all verified numbers for paper update ───────────────────────────────
sep("VERIFIED NUMBERS FOR PAPER")
print(f"ARIMA order:  ({best_p}, 1, {best_q})")
print(f"ARIMA AIC:    {best_aic:.2f}")
print(f"ARIMA BIC:    {best_bic:.2f}")
print(f"VAR lag:      {best_lag}")
print()
print(metrics_df.to_string(index=False))
print()
print("Granger causality:")
print(granger_df.to_string())
print()
print(f"CUSUM boundary breached: {cusum_breach}")
print(f"CUSUM bound (5%): +/- {bound:.6f}")
print(f"Max |CUSUM|: {cusum.abs().max():.6f}")

# Save granger and metrics tables for reference
granger_df.to_csv("outputs/granger_causality.csv")
metrics_df.to_csv("outputs/model_metrics_full.csv", index=False)
print("\nSaved outputs/granger_causality.csv")
print("Saved outputs/model_metrics_full.csv")
