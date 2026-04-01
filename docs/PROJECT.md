# Project Overview

## Research Question
*Can lagged macroeconomic variables — including unemployment, money supply, oil prices, and the federal funds rate — be used to forecast U.S. CPI inflation?*

## Motivation
Inflation forecasting is one of the central problems in macroeconomics. Accurate forecasts inform monetary policy, business planning, and household financial decisions. This project uses publicly available FRED data and standard econometric methods to build, evaluate, and compare forecasting models.

## Data
All data is sourced from the Federal Reserve Economic Data (FRED) API and covers January 2000 to present, resampled to monthly frequency.

| Variable | FRED Series ID | Description |
|---|---|---|
| CPI | CPIAUCSL | Consumer Price Index for All Urban Consumers |
| Unemployment | UNRATE | U.S. Unemployment Rate (%) |
| M2 | M2SL | M2 Money Supply (billions of dollars) |
| Oil | DCOILWTICO | WTI Crude Oil Price (dollars per barrel) |
| Fed Funds | FEDFUNDS | Federal Funds Effective Rate (%) |

## Methodology

### 1. Data Collection & Cleaning
- Pull series from FRED using the fredapi Python library
- Resample to monthly frequency
- Handle missing values

### 2. Exploratory Data Analysis
- Time series plots for each variable
- Correlation matrix
- ACF and PACF plots
- Seasonal decomposition

### 3. Stationarity Testing
- Augmented Dickey-Fuller (ADF) test on each series
- First-difference non-stationary series
- Confirm stationarity of transformed series

### 4. Modeling
- **ARIMA** — univariate model on CPI only; parameters selected via AIC/BIC grid search
- **VAR** — multivariate model using all variables; lag order selected via information criteria

### 5. Evaluation
- 80/20 train/test split
- Out-of-sample forecasting
- Metrics: RMSE, MAE
- Visual comparison of forecast vs. actual

## Deliverables
| Deliverable | Location | Status |
|---|---|---|
| Data pipeline | src/fetch_data.py | 🟡 In Progress |
| EDA notebook | notebooks/eda.ipynb | 🔴 Not Started |
| Modeling notebook | notebooks/models.ipynb | 🔴 Not Started |
| Streamlit dashboard | dashboard/app.py | 🔴 Not Started |
| Research paper | paper/paper.md | 🔴 Not Started |

## References
- Stock, J.H. & Watson, M.W. (2007). *Introduction to Econometrics*
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.) — free at otexts.com/fpp3
- FRED API Documentation — fred.stlouisfed.org/docs/api
