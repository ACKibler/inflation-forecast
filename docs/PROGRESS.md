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
