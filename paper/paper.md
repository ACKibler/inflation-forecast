# Forecasting U.S. Inflation Using Macroeconomic Time Series: An ARIMA and VAR Approach

**Author:** Ashton Kibler  
**Date:** April 2026  
**Data:** Federal Reserve Economic Data (FRED), January 2000 – February 2026  
**Code & Dashboard:** https://github.com/ACKibler/inflation-forecast

---

## Abstract

This paper applies classical econometric time series methods to forecast U.S. consumer price inflation. Using five macroeconomic series sourced from the Federal Reserve Economic Data (FRED) API — the Consumer Price Index (CPI), unemployment rate, M2 money supply, WTI crude oil price, and the federal funds rate — I estimate and compare a univariate Autoregressive Integrated Moving Average (ARIMA) model and a multivariate Vector Autoregression (VAR) model. Both models are evaluated on an out-of-sample test set covering December 2020 through February 2026, a period that includes the most significant inflation surge in four decades. The VAR model outperforms ARIMA across all forecast error metrics, achieving a Mean Absolute Percentage Error (MAPE) of 7.20% versus 9.88% for ARIMA, confirming that macroeconomic covariates carry meaningful information for inflation forecasting beyond the univariate CPI history alone.

---

## 1. Introduction

Inflation — the rate at which the general price level rises — is among the most closely watched variables in macroeconomics. Central banks target it explicitly, businesses use it to plan investment, and households rely on it to preserve purchasing power. Despite its importance, inflation forecasting remains difficult, particularly during structural breaks such as the COVID-19 pandemic and its aftermath.

The two workhorse models of econometric time series forecasting are the Autoregressive Integrated Moving Average (ARIMA) model, introduced by Box and Jenkins (1970), and the Vector Autoregression (VAR) model, popularized by Sims (1980). ARIMA treats CPI as a univariate process governed solely by its own history, while VAR jointly models CPI alongside other macroeconomic variables, allowing cross-variable dynamics to inform the forecast.

This paper addresses the following research question:

> *Can lagged macroeconomic variables — including unemployment, money supply, oil prices, and the federal funds rate — improve forecasts of U.S. CPI inflation relative to a univariate ARIMA benchmark?*

The analysis covers January 2000 through February 2026, spanning the dot-com bust, the 2008 financial crisis, the COVID-19 shock, and the 2021–2023 inflation surge. This time span encompasses multiple business cycles and provides a demanding out-of-sample evaluation environment.

---

## 2. Data

### 2.1 Sources and Series

All data are sourced from the Federal Reserve Economic Data (FRED) API maintained by the Federal Reserve Bank of St. Louis. Five monthly series are used, each beginning in January 2000:

| Variable | FRED ID | Description | Units |
|---|---|---|---|
| CPI | CPIAUCSL | Consumer Price Index, All Urban Consumers | Index (1982–84 = 100) |
| UNRATE | UNRATE | U.S. Unemployment Rate | Percent |
| M2 | M2SL | M2 Money Supply | Billions of dollars |
| OIL | DCOILWTICO | WTI Crude Oil Price | Dollars per barrel |
| FEDFUNDS | FEDFUNDS | Federal Funds Effective Rate | Percent |

The FRED API is accessed programmatically via the `fredapi` Python library. All series are resampled to monthly start frequency (`MS`) using the period mean, ensuring a consistent time index across variables with differing native frequencies (daily for OIL, monthly for the others).

### 2.2 Sample and Coverage

The final dataset contains **314 monthly observations** from January 2000 through February 2026. One observation (October 2025) is absent from FRED for both CPI and UNRATE and is excluded. Months where only secondary series (OIL, M2) are missing are retained via linear interpolation, preserving a gapless monthly index for modeling purposes.

### 2.3 Descriptive Statistics

Table 1 reports summary statistics for all five series over the full sample.

**Table 1: Descriptive Statistics (January 2000 – February 2026)**

| Variable | Mean | Std. Dev. | Min | Max |
|---|---|---|---|---|
| CPI | 233.77 | 41.72 | 169.3 | 327.5 |
| UNRATE (%) | 5.64 | 1.94 | 3.4 | 14.8 |
| M2 (B$) | 11,820 | 5,642 | 4,668 | 22,667 |
| OIL ($/bbl) | 63.72 | 24.84 | 16.6 | 133.9 |
| FEDFUNDS (%) | 2.01 | 2.02 | 0.05 | 6.54 |

Several features of the data are worth noting. CPI and M2 both exhibit strong upward trends over the sample, reflecting secular inflation and monetary expansion. The unemployment rate shows sharp cyclical spikes corresponding to the 2008–2009 recession (peak: 10.0%) and the COVID-19 shock (peak: 14.8%). Oil prices are the most volatile series, ranging from $16.55 per barrel in April 2020 to $133.88 in June 2022. The federal funds rate follows the characteristic step-function pattern of monetary policy: elevated pre-2009, near-zero through 2015, rising through 2019, returning to zero in 2020, then rising sharply from 2022 onward in response to inflation.

The correlation matrix (Figure 1) reveals a near-perfect positive correlation between CPI and M2 (r = 0.97), consistent with the quantity theory of money over long horizons. CPI is negatively correlated with unemployment (r = −0.25), in line with the Phillips curve relationship, though this simple contemporaneous correlation understates the dynamic relationship captured by the VAR.

---

## 3. Methodology

### 3.1 Stationarity Analysis

Time series regression requires stationary inputs. The Augmented Dickey-Fuller (ADF) test is applied to each series in levels, first differences, and log-differences. Results are summarized in Table 2.

**Table 2: ADF Test Results**

| Series | Level p-value | Stationary? | First Diff p-value | Stationary? | Treatment |
|---|---|---|---|---|---|
| CPI | 0.997 | No | 0.080 | No | Log-difference |
| UNRATE | 0.036 | Yes | — | — | Levels |
| M2 | 0.990 | No | 0.017 | Yes | Log-difference |
| OIL | 0.030 | Yes | — | — | Levels |
| FEDFUNDS | 0.001 | Yes | — | — | Levels |

CPI is integrated of order greater than one in its level form, meaning a simple first difference is insufficient for stationarity (p = 0.080, failing the 5% threshold). The log-difference of CPI — equivalent to the continuously compounded monthly inflation rate — achieves stationarity (p = 0.019). This transformation is also economically meaningful: `d(log(CPI))` approximates the month-over-month percentage change in prices, which is the object of direct policy interest. M2 is treated analogously.

### 3.2 ARIMA Model

The ARIMA(p, d, q) model for log-differenced CPI is estimated as:

$$\Delta \log(\text{CPI}_t) = \mu + \sum_{i=1}^{p} \phi_i \Delta \log(\text{CPI}_{t-i}) + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}$$

where d = 1 is imposed (equivalently, the model is fit on log(CPI) with one difference). The orders p and q are selected via a grid search over p, q ∈ {0, 1, 2, 3, 4}, minimizing the Akaike Information Criterion (AIC). The AIC-optimal specification is **ARIMA(1, 1, 4)** with AIC = −2251.70.

Residual diagnostics confirm model adequacy. The Ljung-Box test fails to reject the null of no autocorrelation at lags 10 and 20 (p ≈ 1.0 in both cases), indicating the model has extracted the systematic serial dependence from the series. The residual ACF and PACF are flat within confidence bands.

### 3.3 VAR Model

The Vector Autoregression jointly models the vector of stationary-transformed variables:

$$\mathbf{y}_t = [\Delta\log(\text{CPI}_t),\ \Delta \text{UNRATE}_t,\ \Delta\log(\text{M2}_t),\ \text{OIL}_t,\ \text{FEDFUNDS}_t]'$$

The VAR(p) model is:

$$\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t$$

where **c** is a vector of intercepts, **A**₁…**A**ₚ are 5×5 coefficient matrices, and **ε**ₜ is a white noise vector with covariance matrix **Σ**. The lag order p is selected by minimizing AIC over p ∈ {1, …, 12}, yielding **VAR(4)** (AIC = −25.59).

The estimated DCPI equation shows several statistically significant predictors beyond its own lags. Lagged OIL (L1, p < 0.001) and lagged FEDFUNDS (L1, p = 0.004; L4, p = 0.019) are positive predictors of inflation, consistent with cost-push and monetary transmission channels. Lagged DUNRATE (L2, L3) enters negatively, reflecting the Phillips curve: rising unemployment precedes falling inflation.

Residual serial correlation is assessed using Durbin-Watson statistics, which are close to 2.0 for all five equations (range: 1.94–2.04), indicating no significant autocorrelation in the VAR residuals.

### 3.4 Impulse Response Functions

Orthogonalised Impulse Response Functions (IRFs) trace the dynamic effect of a one-standard-deviation shock to each variable on all others over a 24-month horizon. Key findings:

- **OIL → DCPI:** A positive oil price shock produces a persistent positive response in CPI inflation, peaking at approximately 3 months and decaying by month 12. This is consistent with the empirical literature on oil price pass-through to consumer prices.
- **FEDFUNDS → DUNRATE:** A positive federal funds rate shock (monetary tightening) produces a delayed negative response in the change in unemployment — i.e., unemployment rises — with the effect materializing over a 6–12 month horizon, consistent with the conventional monetary policy transmission lag.
- **DM2 → DUNRATE:** An expansionary money supply shock produces a short-run positive response in unemployment change, reflecting the complexity of the money-unemployment relationship over the business cycle.

### 3.5 Evaluation Framework

Both models are evaluated on an identical train/test split:

- **Training set:** January 2000 – November 2020 (250 observations)
- **Test set:** December 2020 – February 2026 (63 observations)

The test period deliberately includes the 2021–2023 inflation surge — a severe out-of-sample stress test for any model estimated on pre-pandemic data. Forecasts are generated recursively for the full test horizon without re-estimation. For the VAR, the DCPI forecast is back-transformed to CPI levels via:

$$\widehat{\text{CPI}}_t = \exp\!\left(\log(\text{CPI}_T) + \sum_{s=T+1}^{t} \widehat{\Delta\log(\text{CPI}_s)}\right)$$

where T is the last training observation. Three error metrics are reported: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

---

## 4. Results

### 4.1 Forecast Accuracy

Table 3 reports out-of-sample forecast errors for both models over the December 2020 – February 2026 test period.

**Table 3: Out-of-Sample Forecast Error (Test Set: Dec 2020 – Feb 2026)**

| Model | RMSE | MAE | MAPE (%) |
|---|---|---|---|
| ARIMA(1,1,4) | 33.74 | 30.57 | 9.88% |
| VAR(4) | **24.64** | **22.28** | **7.20%** |

The VAR model outperforms ARIMA by approximately 27% on both RMSE and MAE, and reduces MAPE by 2.7 percentage points. This improvement is consistent across all three metrics, indicating a robust performance advantage rather than a metric-specific artifact.

### 4.2 Forecast Comparison

Figure 2 overlays both model forecasts against the actual CPI path. Both models correctly project a rising CPI trend, reflecting the persistent inflationary momentum embedded in the pre-pandemic training data. However, neither model fully anticipates the sharp acceleration in prices that began in mid-2021. This is expected: the magnitude of the post-pandemic inflation surge — driven by unprecedented fiscal stimulus, supply chain disruptions, and energy price shocks — was without precedent in the training sample.

The VAR forecast tracks the actual CPI more closely in the early test period (2021–2022) before diverging, whereas the ARIMA forecast flattens more aggressively as the forecast horizon extends. The absolute error plot (Figure 2, lower panel) shows that VAR errors are consistently smaller than ARIMA errors throughout the test window, with the gap widening after 2022.

### 4.3 Discussion

The VAR's advantage over ARIMA can be attributed to two mechanisms. First, the FEDFUNDS and OIL variables carry contemporaneous and lagged information about inflationary pressure that is not captured by CPI history alone. The statistically significant coefficients on lagged OIL and FEDFUNDS in the DCPI equation confirm that these series Granger-cause inflation within the model. Second, the VAR's multivariate structure allows it to incorporate the monetary policy tightening cycle that began in 2022 (rising FEDFUNDS), partially offsetting the inflationary impulse in its forecasts.

That said, both models significantly underforecast the peak inflation of 2022. This reflects a fundamental limitation of linear time series models estimated on stable-regime data: they cannot anticipate regime changes of the magnitude seen post-2020. More flexible methods — such as time-varying parameter VARs, Bayesian VARs with stochastic volatility, or machine learning approaches — may perform better in such environments, at the cost of greater complexity and reduced interpretability.

---

## 5. Conclusion

This paper compares ARIMA and VAR models for forecasting U.S. CPI inflation using publicly available FRED data from January 2000 through February 2026. The main findings are:

1. **CPI requires log-differencing** to achieve stationarity; a simple first difference is insufficient due to the series' exponential growth trend.
2. **The VAR(4) model outperforms ARIMA(1,1,4)** across all forecast error metrics on the December 2020 – February 2026 test set (MAPE: 7.20% vs. 9.88%), confirming that macroeconomic covariates — particularly oil prices and the federal funds rate — carry incremental predictive information for inflation.
3. **Neither model fully captures the 2021–2023 inflation surge**, reflecting the structural break caused by the pandemic and its policy response. This highlights the limits of linear, fixed-parameter models during periods of macroeconomic instability.

These results suggest that practitioners should prefer multivariate approaches over univariate benchmarks when forecasting inflation, particularly when policy variables (FEDFUNDS) and cost-push indicators (OIL) are available. Future work could extend this analysis by incorporating time-varying parameters, Bayesian shrinkage priors, or hybrid machine learning approaches to improve robustness during structural breaks.

---

## References

Box, G.E.P., & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.

Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3

Sims, C.A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1–48.

Stock, J.H., & Watson, M.W. (2007). *Introduction to Econometrics* (2nd ed.). Pearson.

Stock, J.H., & Watson, M.W. (2001). Vector Autoregressions. *Journal of Economic Perspectives*, 15(4), 101–115.

Federal Reserve Bank of St. Louis. (2026). *Federal Reserve Economic Data (FRED)*. https://fred.stlouisfed.org

McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96.
