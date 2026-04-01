# Task Board

## 🔴 Backlog

### Data
- [ ] Verify fetch_data.py runs and saves macro_data.csv cleanly
- [ ] Add data validation checks (missing values, date ranges)
- [ ] Document each FRED series in a data dictionary

### EDA
- [ ] Create EDA notebook in /notebooks
- [ ] Plot each time series over time
- [ ] Build ACF/PACF plots
- [ ] Run trend decomposition (seasonal, trend, residual)
- [ ] Compute summary statistics table

### Stationarity & Preprocessing
- [ ] Run ADF tests on all series
- [ ] Difference non-stationary series
- [ ] Re-run ADF tests on differenced series
- [ ] Log-transform skewed variables if needed

### Modeling
- [ ] Fit ARIMA model on CPI
- [ ] Select optimal ARIMA(p,d,q) parameters using AIC/BIC
- [ ] Fit VAR model with all variables
- [ ] Select optimal VAR lag order
- [ ] Generate impulse response functions (IRF)

### Evaluation
- [ ] Split data into train/test sets (80/20)
- [ ] Generate out-of-sample forecasts
- [ ] Compute RMSE and MAE for each model
- [ ] Compare ARIMA vs VAR performance
- [ ] Plot forecast vs actual values

### Dashboard
- [ ] Scaffold Streamlit app in /dashboard
- [ ] Add time series visualization panel
- [ ] Add forecast visualization panel
- [ ] Add model comparison panel
- [ ] Deploy dashboard (Streamlit Cloud)

### Paper
- [ ] Write introduction and research question
- [ ] Write data section
- [ ] Write methodology section
- [ ] Write results section
- [ ] Write conclusion
- [ ] Add references

---

## 🟡 In Progress
- [ ] Verify fetch_data.py runs and saves macro_data.csv

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
