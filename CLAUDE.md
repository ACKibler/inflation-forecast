# CLAUDE.md — Project Context File
> This file is the single source of truth for both Claude Code (VS Code) and Claude.ai.
> Update this file at the end of every session and commit it to GitHub.

---

## Project Summary
A Python-based econometrics research project forecasting U.S. inflation using macroeconomic
time series data from the FRED API. The goal is to build, evaluate, and compare ARIMA and VAR
forecasting models, present findings in a Streamlit dashboard, and document the work in a
research paper.

**Repo:** https://github.com/ACKibler/inflation-forecast
**Owner:** Ashton Kibler
**Started:** 2026-03-31

---

## Project Structure
```
inflation-forecast/
├── data/               # Raw + cleaned CSVs from FRED
├── notebooks/          # Jupyter notebooks for EDA and modeling
│   └── eda.ipynb       # Fully executed EDA notebook
├── src/                # Python scripts
│   ├── fetch_data.py   # Pulls FRED data and saves to data/macro_data.csv
│   ├── validate_data.py# Data validation checks (nulls, date range, frequency, ranges)
│   ├── plot_missing.py # Visualizes missing data across all series
│   └── stationarity.py # ADF tests on levels, first diffs, and log-diffs
├── outputs/            # All saved plot PNGs
├── dashboard/          # Streamlit app (not yet started)
├── paper/              # Research writeup (not yet started)
├── docs/
│   ├── PROJECT.md      # Full methodology and research overview
│   ├── TASKS.md        # Kanban task board
│   ├── PROGRESS.md     # Weekly progress log
│   └── DECISIONS.md    # Key decision log
├── CLAUDE.md           # This file — shared context for Claude tools
├── .env                # FRED_API_KEY (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack
| Purpose | Library |
|---|---|
| Data pulling | fredapi |
| Data manipulation | pandas, numpy |
| Modeling | statsmodels, scikit-learn |
| Visualization | matplotlib, seaborn |
| Dashboard | streamlit |
| Environment | python-dotenv |
| Python version | 3.10 |

---

## Data
All data from FRED API, January 2000 to present, resampled to monthly frequency.
Saved to `data/macro_data.csv` (313 rows × 5 columns as of 2026-03-31).

| Variable | FRED ID | Description |
|---|---|---|
| CPI | CPIAUCSL | Consumer Price Index |
| UNRATE | UNRATE | Unemployment Rate |
| M2 | M2SL | M2 Money Supply |
| OIL | DCOILWTICO | WTI Crude Oil Price |
| FEDFUNDS | FEDFUNDS | Federal Funds Rate |

**Known data issue:** October 2025 is missing for both CPI and UNRATE on FRED (not a pipeline
bug — the data simply isn't published there). `fetch_data.py` uses `dropna(subset=["CPI","UNRATE"])`
so months where only secondary series (OIL, M2) are late are retained.

---

## Stationarity Findings
ADF tests run in `src/stationarity.py`. Results inform modeling treatment:

| Series | Level | First Diff | Treatment for modeling |
|---|---|---|---|
| CPI | Non-stationary | Non-stationary | **Log-difference** (≈ monthly inflation rate) |
| UNRATE | Stationary | — | Use levels |
| M2 | Non-stationary | Stationary | First difference |
| OIL | Stationary | — | Use levels |
| FEDFUNDS | Stationary | — | Use levels |

---

## Current Status
🟡 **Week 1 — Infrastructure, EDA, and Stationarity complete**

### Recently Completed
- [x] fetch_data.py verified: pulls 313 months of data, saves to data/macro_data.csv
- [x] fredapi installed, dropna fixed to only drop on CPI/UNRATE
- [x] src/validate_data.py: 5 checks (columns, date range, frequency gaps, nulls, value ranges)
- [x] src/plot_missing.py: missing data heatmap saved to outputs/missing_data.png
- [x] notebooks/eda.ipynb: fully executed — time series plots, correlation heatmap, ACF/PACF, seasonal decomposition (13 output PNGs in outputs/)
- [x] src/stationarity.py: ADF tests on levels, first differences, and log-differences
- [x] CPI confirmed to need log-differencing; all series stationary after treatment
- [x] src/arima_model.py: AIC grid search → ARIMA(1,1,4), RMSE=33.74, MAE=30.57, MAPE=9.88%
- [x] src/var_model.py: VAR(4) via AIC lag selection, IRF, RMSE=24.64, MAE=22.28, MAPE=7.20%
- [x] src/evaluate_models.py: side-by-side comparison, overlay plot, bar chart, model_metrics.csv
- [x] dashboard/app.py: 4-tab Streamlit app (time series, forecasts, comparison, about)

### In Progress
- [ ] None

### Next Up
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Write paper sections (intro, data, methodology, results, conclusion)

---

## Key Conventions
- **Commit prefixes:** `docs:` `feat:` `fix:` `data:` `model:`
- **Credentials:** Always in .env, never committed
- **Documentation:** Update TASKS.md, PROGRESS.md, and this file at end of every session
- **Branching:** Work on main for now, introduce feature branches when modeling begins

---

## Known Issues & Notes
- Python installed via Microsoft Store — scripts PATH warning appears but is non-blocking
- October 2025 missing from FRED for CPI and UNRATE — accepted as a data gap, not a bug
- Windows terminal uses cp1252 encoding — avoid Unicode arrows (→) in print statements

---

## How to Use This File
**Claude Code:** This file is auto-read at session start. You are fully oriented.
**Claude.ai:** Paste the contents of this file at the start of a new conversation to restore full context.

### End of Session Checklist
Ask Claude Code to run these steps before closing:
1. Update `Current Status` section in this file
2. Move completed tasks in `docs/TASKS.md`
3. Add entry to `docs/PROGRESS.md`
4. Log any new decisions in `docs/DECISIONS.md`
5. Commit and push:
```bash
git add .
git commit -m "docs: end of session update"
git push
```
