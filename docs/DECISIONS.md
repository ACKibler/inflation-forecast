# Decision Log

A running log of key project decisions and the reasoning behind them.

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
