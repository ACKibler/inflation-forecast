"""
Inflation Forecast Dashboard
Run with: streamlit run dashboard/app.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from fredapi import Fred
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="U.S. Inflation Forecast",
    page_icon="📈",
    layout="wide",
)

st.title("U.S. Inflation Forecast")
st.caption("ARIMA vs VAR models trained on FRED macroeconomic data (2000 – present)")

# ── load & cache data ─────────────────────────────────────────────────────────
def _get_api_key():
    """Return FRED API key from st.secrets (cloud) or .env (local)."""
    # Try Streamlit secrets first (cloud deployment)
    try:
        key = st.secrets["FRED_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass
    # Fall back to .env for local development
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("FRED_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # Neither source worked
    st.error(
        "**FRED API key not found.**\n\n"
        "On Streamlit Cloud: go to **Manage app → Settings → Secrets** and add:\n"
        "```toml\nFRED_API_KEY = \"your_key_here\"\n```"
    )
    st.stop()

@st.cache_data(show_spinner="Fetching data from FRED...")
def load_data():
    # Use local CSV if available (faster for development)
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "macro_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        api_key = _get_api_key()
        fred = Fred(api_key=api_key)
        series = {
            "CPI":      "CPIAUCSL",
            "UNRATE":   "UNRATE",
            "M2":       "M2SL",
            "OIL":      "DCOILWTICO",
            "FEDFUNDS": "FEDFUNDS",
        }
        dfs = []
        for name, sid in series.items():
            data = fred.get_series(sid, observation_start="2000-01-01")
            data.name = name
            dfs.append(data)
        df = pd.concat(dfs, axis=1).resample("MS").mean()
        df.dropna(subset=["CPI", "UNRATE"], inplace=True)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(full_idx).interpolate(method="time")
    df.index.freq = "MS"
    return df

@st.cache_data
def run_arima(df, split):
    log_cpi     = np.log(df["CPI"])
    train       = log_cpi.iloc[:split]
    test        = log_cpi.iloc[split:]
    grid = []
    for p, q in product(range(5), range(5)):
        try:
            m = ARIMA(train, order=(p, 1, q)).fit()
            grid.append((m.aic, p, q))
        except Exception:
            pass
    _, best_p, best_q = min(grid)
    fitted   = ARIMA(train, order=(best_p, 1, best_q)).fit()
    fc       = np.exp(fitted.get_forecast(steps=len(test)).predicted_mean)
    fc_ci    = fitted.get_forecast(steps=len(test)).conf_int(alpha=0.05)
    fc_lower = np.exp(fc_ci.iloc[:, 0])
    fc_upper = np.exp(fc_ci.iloc[:, 1])
    actual   = np.exp(test)
    return best_p, best_q, fc, fc_lower, fc_upper, actual

@st.cache_data
def run_var(df, split):
    stat = pd.DataFrame(index=df.index)
    stat["DCPI"]     = np.log(df["CPI"]).diff()
    stat["DUNRATE"]  = df["UNRATE"].diff()
    stat["DM2"]      = np.log(df["M2"]).diff()
    stat["OIL"]      = df["OIL"]
    stat["FEDFUNDS"] = df["FEDFUNDS"]
    stat.dropna(inplace=True)

    train    = stat.iloc[:split]
    test     = stat.iloc[split:]
    best_lag = VAR(train).select_order(maxlags=12).selected_orders["aic"]
    fitted   = VAR(train).fit(best_lag)
    fc_arr   = fitted.forecast(train.values[-best_lag:], steps=len(test))
    fc_df    = pd.DataFrame(fc_arr, index=test.index, columns=stat.columns)

    last_log = np.log(df["CPI"]).iloc[split]
    fc_cpi   = np.exp(last_log + fc_df["DCPI"].cumsum())
    actual   = df["CPI"].iloc[split + 1 : split + 1 + len(fc_cpi)]
    min_len  = min(len(fc_cpi), len(actual))
    return best_lag, fc_cpi.iloc[:min_len], actual.iloc[:min_len]

def calc_metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae  = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return rmse, mae, mape

# ── sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")
train_pct = st.sidebar.slider("Train split (%)", 60, 90, 80, step=5)

df   = load_data()
split = int(len(df) * train_pct / 100)

with st.spinner("Running models (this takes ~30 seconds on first load)..."):
    best_p, best_q, fc_arima, fc_arima_lo, fc_arima_hi, actual_arima = run_arima(df, split)
    best_lag, fc_var, actual_var = run_var(df, split)

arima_rmse, arima_mae, arima_mape = calc_metrics(actual_arima, fc_arima)
var_rmse,   var_mae,   var_mape   = calc_metrics(actual_var,   fc_var)

arima_label = f"ARIMA({best_p},1,{best_q})"
var_label   = f"VAR({best_lag})"

# ── tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Time Series", "🔮 Forecasts", "📉 Model Comparison", "ℹ️ About"
])

# ── Tab 1: Raw time series ────────────────────────────────────────────────────
with tab1:
    st.subheader("Macroeconomic Series (2000 – Present)")
    series_labels = {
        "CPI":      "Consumer Price Index",
        "UNRATE":   "Unemployment Rate (%)",
        "M2":       "M2 Money Supply (B$)",
        "OIL":      "WTI Crude Oil ($/bbl)",
        "FEDFUNDS": "Federal Funds Rate (%)",
    }
    selected = st.multiselect("Select series", list(series_labels.keys()),
                              default=["CPI", "FEDFUNDS", "UNRATE"])
    for col in selected:
        fig = px.line(df, x=df.index, y=col,
                      title=series_labels[col],
                      labels={"x": "Date", col: series_labels[col]})
        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Forecasts ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("CPI Forecast vs Actual")
    model_choice = st.radio("Model", [arima_label, var_label], horizontal=True)

    train_cpi = df["CPI"].iloc[:split + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_cpi.index, y=train_cpi,
                             name="Train", line=dict(color="#1f77b4", width=1.5)))

    if model_choice == arima_label:
        fig.add_trace(go.Scatter(x=actual_arima.index, y=actual_arima,
                                 name="Actual", line=dict(color="#2ca02c", width=2)))
        fig.add_trace(go.Scatter(x=fc_arima.index, y=fc_arima,
                                 name=arima_label, line=dict(color="#d62728", width=2, dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_arima_hi, fc_arima_lo.iloc[::-1]]).index,
            y=pd.concat([fc_arima_hi, fc_arima_lo.iloc[::-1]]).values,
            fill="toself", fillcolor="rgba(214,39,40,0.1)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI"
        ))
        rmse, mae, mape = arima_rmse, arima_mae, arima_mape
    else:
        fig.add_trace(go.Scatter(x=actual_var.index, y=actual_var,
                                 name="Actual", line=dict(color="#2ca02c", width=2)))
        fig.add_trace(go.Scatter(x=fc_var.index, y=fc_var,
                                 name=var_label, line=dict(color="#d62728", width=2, dash="dot")))
        rmse, mae, mape = var_rmse, var_mae, var_mape

    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="CPI",
                      title=f"{model_choice} — CPI Forecast")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE",  f"{mae:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")

# ── Tab 3: Model comparison ───────────────────────────────────────────────────
with tab3:
    st.subheader("ARIMA vs VAR — Head to Head")

    # Overlay plot
    train_cpi = df["CPI"].iloc[:split + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_cpi.index, y=train_cpi,
                             name="Train", line=dict(color="#1f77b4", width=1.2)))
    fig.add_trace(go.Scatter(x=actual_arima.index, y=actual_arima,
                             name="Actual", line=dict(color="#2ca02c", width=2.5)))
    fig.add_trace(go.Scatter(x=fc_arima.index, y=fc_arima,
                             name=arima_label, line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=fc_var.index, y=fc_var,
                             name=var_label, line=dict(color="#d62728", width=2, dash="dot")))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="CPI",
                      title="Both Forecasts vs Actual CPI")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        "Model": [arima_label, var_label],
        "RMSE":     [round(arima_rmse, 3), round(var_rmse, 3)],
        "MAE":      [round(arima_mae, 3),  round(var_mae, 3)],
        "MAPE (%)": [round(arima_mape, 2), round(var_mape, 2)],
    }).set_index("Model")
    st.dataframe(metrics_df.style.highlight_min(axis=0, color="#c6efce"), use_container_width=True)

    # Bar charts
    cols = st.columns(3)
    for col, metric in zip(cols, ["RMSE", "MAE", "MAPE (%)"]):
        fig_bar = px.bar(
            metrics_df.reset_index(), x="Model", y=metric,
            color="Model", color_discrete_sequence=["#ff7f0e", "#d62728"],
            title=metric, text_auto=".2f",
        )
        fig_bar.update_layout(height=280, showlegend=False,
                              margin=dict(t=40, b=10))
        col.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 4: About ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("About this project")
    st.markdown("""
**Goal:** Forecast U.S. CPI inflation using classical econometric time series models.

**Data:** Five macroeconomic series from the FRED API (2000 – present), resampled to monthly frequency.

| Series | FRED ID | Description |
|---|---|---|
| CPI | CPIAUCSL | Consumer Price Index |
| UNRATE | UNRATE | Unemployment Rate |
| M2 | M2SL | M2 Money Supply |
| OIL | DCOILWTICO | WTI Crude Oil Price |
| FEDFUNDS | FEDFUNDS | Federal Funds Rate |

**Stationarity treatment:**
- CPI → log-differenced (monthly inflation rate)
- M2 → log-differenced
- UNRATE → first-differenced
- OIL, FEDFUNDS → used in levels (ADF confirmed stationary)

**Models:**
- **ARIMA** — univariate benchmark on log(CPI), order selected by AIC grid search over p,q ∈ {0..4}
- **VAR** — multivariate model with all five series, lag selected by AIC up to 12 lags

**Author:** Ashton Kibler | [GitHub](https://github.com/ACKibler/inflation-forecast)
""")
