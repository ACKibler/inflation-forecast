# Decision Log

A running log of key project decisions and the reasoning behind them.

---

## 2026-04-01

**Decision:** Correct VAR MAPE from 7.20% to 7.55% in paper
**Reason:** Re-running paper_analysis.py revealed the earlier evaluate_models.py used a slightly different index alignment when back-transforming VAR log-differences to CPI levels, yielding a marginally different test set. The corrected figure uses a consistent SPLIT=int(len(df)*0.8) index throughout.

---

**Decision:** Use recursive (non-rolling) out-of-sample forecasting
**Reason:** Model estimated once on training data and projected forward 63 steps without re-estimation. This is more conservative and susceptible to parameter instability but is standard for fixed-parameter linear models. Acknowledged in Section 3.7 with a note that rolling window evaluation is a natural extension.

---

**Decision:** Include FEDFUNDS in levels despite ZLB behavior
**Reason:** Full-sample ADF rejects unit root at 1% level. Economic theory precludes a permanent unit root for a policy rate. ZLB behavior is acknowledged in a footnote in Section 3.1 for transparency.

---

**Decision:** Use CUSUM test (not Chow test) for stability analysis
**Reason:** CUSUM does not require a pre-specified break date, making it more appropriate when the break (COVID-19 onset) may affect residuals gradually rather than at a single known date. The Chow test would require specifying March or April 2020 as the break point a priori.

---

## 2026-03-31

**Decision:** Use FRED API over manual CSV downloads
**Reason:** Programmatic access ensures reproducibility. Anyone cloning the repo can pull the same data with a single script rather than manually downloading files.

---

**Decision:** Resample all series to monthly frequency
**Reason:** FRED series have different native frequencies (daily, monthly, quarterly). Resampling to monthly using `.resample("MS").mean()` gives a consistent time index across all variables.

---

**Decision:** Start data from 2000-01-01
**Reason:** Captures multiple business cycles (dot-com bust, 2008 financial crisis, COVID-19, post-COVID inflation surge) while keeping the dataset manageable. Pre-2000 data introduces structural breaks that complicate beginner-level modeling.

---

**Decision:** Use ARIMA and VAR as primary models
**Reason:** ARIMA is the standard univariate benchmark for time series forecasting. VAR extends this to the multivariate case and is the workhorse of macroeconomic forecasting in the literature. Both are well-supported in statsmodels and have strong pedagogical value.

---

**Decision:** Store API key in .env file
**Reason:** Keeps credentials out of version control. The .env file is listed in .gitignore so it is never pushed to GitHub.

---

**Decision:** Use Streamlit for the dashboard
**Reason:** Python-native, minimal boilerplate, easy to deploy. Allows the dashboard to be built in the same language as the rest of the project without learning a separate framework.

---

**Decision:** Manage project via Markdown files tracked in Git
**Reason:** Keeps all project documentation version-controlled alongside the code. Provides a clear audit trail of decisions, progress, and tasks without needing external project management tools.
