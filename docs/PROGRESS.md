# Progress Log

A weekly log of work completed, blockers encountered, and next steps.

---

## Week 1 — 2026-03-31

### ✅ Completed
- Set up GitHub repository and connected to VS Code
- Initialized project folder structure (data, notebooks, src, dashboard, paper, docs)
- Installed all Python dependencies and generated requirements.txt
- Configured .gitignore to exclude venv/, .env, __pycache__
- Obtained FRED API key
- Wrote src/fetch_data.py to pull CPI, UNRATE, M2, OIL, FEDFUNDS from FRED
- Set up project documentation (README, TASKS, DECISIONS, PROGRESS)

### 🚧 Blockers
- FRED API key had a leading space in .env causing a ValueError — resolved by removing the space
- Python installed via Microsoft Store caused PATH warning for scripts — noted, not blocking

### 🔜 Next Steps
- Verify fetch_data.py runs successfully and macro_data.csv is saved to /data
- Create EDA notebook in /notebooks
- Begin plotting time series and building ACF/PACF plots

---

## Week 1 (continued) — 2026-03-31

### ✅ Completed
- Verified fetch_data.py runs end-to-end; data/macro_data.csv saved (313 rows x 5 cols)
- Installed fredapi; fixed dropna to only require CPI and UNRATE to be non-null
- Wrote src/validate_data.py with 5 checks: columns, date range, frequency gaps, nulls, value ranges
- Wrote src/plot_missing.py; identified October 2025 as the only gap (CPI+UNRATE missing on FRED)
- Created and fully executed notebooks/eda.ipynb: time series plots, correlation heatmap, ACF/PACF, seasonal decomposition — 13 output PNGs saved to outputs/
- Wrote src/stationarity.py; ran ADF tests on levels, first diffs, and log-diffs
- Key finding: CPI requires log-differencing; all series stationary after appropriate treatment

### 🚧 Blockers
- October 2025 missing for CPI and UNRATE on FRED — accepted as a known data gap, not fixable

### 🔜 Next Steps
- Fit ARIMA model on log-differenced CPI
- Grid search for optimal ARIMA(p,d,q) using AIC/BIC
- Fit VAR model with all variables
- Select optimal VAR lag order and generate IRFs

---

## Week 2 — 2026-04-01

### ✅ Completed
- ARIMA(1,1,4) fit and evaluated: RMSE=33.74, MAE=30.57, MAPE=9.88%
- VAR(4) fit and evaluated: RMSE=25.64, MAE=23.36, MAPE=7.55%
- src/evaluate_models.py: overlay comparison plot, bar charts, model_metrics.csv
- dashboard/app.py: 4-tab Streamlit app deployed at https://inflation-forecast-vibe.streamlit.app/
- paper/paper.md: full research paper written (all sections)
- paper/paper.md: critique addressed — numbers verified, Granger tests, random walk benchmark, CUSUM test, figures embedded, FEDFUNDS footnote, Phillips curve expansion, rolling window justification
- src/paper_analysis.py: single script reproducing all paper statistics and generating /figures

### 🚧 Blockers
- VAR MAPE corrected from 7.20% to 7.55% after re-verification (back-transform index alignment)
- CUSUM test detected parameter instability in ARIMA residuals (2020 COVID break) — documented and discussed in paper

### 🔜 Next Steps
- Project is substantially complete
- Optional: rolling window evaluation, Bayesian VAR extension

---

<!-- Copy and paste the block below each week -->
<!--
## Week N — YYYY-MM-DD

### ✅ Completed
-

### 🚧 Blockers
-

### 🔜 Next Steps
-
-->
