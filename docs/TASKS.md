# Task Board

## 🔴 Backlog

### Data
- [x] Verify fetch_data.py runs and saves macro_data.csv cleanly
- [x] Add data validation checks (missing values, date ranges)
- [x] Document each FRED series in a data dictionary (covered in CLAUDE.md data table)

### EDA
- [x] Create EDA notebook in /notebooks
- [x] Plot each time series over time
- [x] Build ACF/PACF plots
- [x] Run trend decomposition (seasonal, trend, residual)
- [x] Compute summary statistics table

### Stationarity & Preprocessing
- [x] Run ADF tests on all series
- [x] Difference non-stationary series
- [x] Re-run ADF tests on differenced series
- [x] Log-transform skewed variables if needed

### Modeling
- [x] Fit ARIMA model on CPI
- [x] Select optimal ARIMA(p,d,q) parameters using AIC/BIC
- [x] Fit VAR model with all variables
- [x] Select optimal VAR lag order
- [x] Generate impulse response functions (IRF)

### Evaluation
- [x] Split data into train/test sets (80/20)
- [x] Generate out-of-sample forecasts
- [x] Compute RMSE and MAE for each model
- [x] Compare ARIMA vs VAR performance
- [x] Plot forecast vs actual values

### Dashboard
- [x] Scaffold Streamlit app in /dashboard
- [x] Add time series visualization panel
- [x] Add forecast visualization panel
- [x] Add model comparison panel
- [x] Deploy dashboard (Streamlit Cloud) — live at https://inflation-forecast-vibe.streamlit.app/

### Paper
- [x] Write introduction and research question
- [x] Write data section
- [x] Write methodology section
- [x] Write results section
- [x] Write conclusion
- [x] Add references
- [x] Verify all numbers against model runs
- [x] Add Granger causality table (Table 4)
- [x] Add naive random walk benchmark (Table 3)
- [x] Run CUSUM stability test and report findings
- [x] Generate and embed all figures (/figures)
- [x] Add FEDFUNDS ZLB footnote
- [x] Expand Phillips curve discussion
- [x] Add rolling window justification paragraph

---

## 🟡 In Progress
- [x] Investigate missing October 2025 row — resolved: genuine FRED reporting lag, handled via dropna(subset=["CPI","UNRATE"]) + time interpolation before modeling

---

## 🟢 Done
- [x] Set up GitHub repository
- [x] Initialize project folder structure
- [x] Install Python dependencies
- [x] Add requirements.txt
- [x] Set up .gitignore
- [x] Obtain FRED API key
- [x] Write fetch_data.py
- [x] Set up project documentation (README, TASKS, DECISIONS, PROGRESS)
