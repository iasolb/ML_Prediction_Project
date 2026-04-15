# Project Summary Reference
## Macroeconomic Forecasting of U.S. Wage Growth — Ian Solberg, ECON 3916

---

### Prediction Question
Can observable macroeconomic indicators predict the year-over-year growth rate of U.S. average weekly earnings?

### Stakeholder
Monetary policy analysts at the Federal Reserve or labor economists at BLS — enables decisions about whether current macro conditions signal upcoming wage acceleration or deceleration.

### Central Finding
**Macro indicators have strong contemporaneous correlation with wage growth but limited out-of-sample predictive power.** Under temporally honest evaluation, all models produce negative R², meaning they predict worse than a naive historical-mean baseline.

---

### Data

| Attribute | Value |
|-----------|-------|
| Source | FRED API via custom FRED_Loader module |
| Raw series | 229 economic time series |
| Scoring layer | 56 blocks → 75+ derived columns |
| Research subset | 34 columns, 998 weekly observations |
| Date range | March 2007 – April 2026 |
| Target | `Avg_Weekly_Earnings_YoY` — YoY % change in avg weekly earnings (BLS CES0500000011) |
| Missing data | 0% in research subset (forward-fill pipeline handles gaps upstream) |

**Target distribution:** Centered ~2–4%, COVID-era spike to ~7.5%, minimum ~0.7%.

---

### Final Feature Set (12 features)

| Feature | Type | Description |
|---------|------|-------------|
| Core_PCE_YoY | Scored | Core PCE inflation, 52-week % change |
| JOLTS_UE_Ratio | Scored | Job openings per unemployed person |
| Personal_Savings_Rate | FRED | Household savings as % of disposable income |
| Labor_Force_Participation | FRED | Civilian labor force participation rate |
| Sahm_Indicator | Scored | 3-month UE avg minus 12-month UE min |
| HY_OAS_Spread | FRED | High-yield corporate bond OAS spread |
| U6_Underemployment | FRED | Broad underemployment rate |
| Employment_Population_Ratio | FRED | Employed persons / civilian population |
| Payroll_Change_4w | Scored | 4-week change in nonfarm payrolls |
| Activity_Momentum | Scored | Composite z-score of IP, retail, PCE, payrolls |
| Earnings_YoY_Lag4 | Derived | Target lagged 4 weeks |
| Earnings_YoY_Lag13 | Derived | Target lagged 13 weeks |

**Feature selection notes:** 3 originally selected features (Retail_Sales, JOLTS_Quits, Part_Time_Economic_Reasons) were replaced with stationary alternatives after temporal evaluation showed trending level variables cause extrapolation failure.

---

### Train/Test Split
- **Temporal split at 80/20:** ~788 train (up to July 2022), ~197 test (July 2022 – April 2026)
- **Rationale:** Ensures model is evaluated on a genuinely future macro regime (post-pandemic wage deceleration)
- `random_state=42` used throughout

---

### Models

| Model | Key Parameters | Description |
|-------|---------------|-------------|
| OLS | None | Closed-form baseline, full interpretability |
| Ridge | α = 1.0 | L2 penalty, shrinks coefficients toward zero |
| GBR | η=0.03, max_depth=3, n_estimators=200, subsample=0.8 | Sequential ensemble of shallow trees fitting residuals |

---

### Results — Test Set (Temporal Split, with lag features)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| OLS | 0.3977 | 0.3169 | 0.011 |
| Ridge | 0.3965 | 0.3162 | 0.017 |
| GBR | 0.4525 | 0.3763 | −0.280 |

### Results — TimeSeriesSplit CV (5-fold)

| Model | CV R² (mean ± std) |
|-------|---------------------|
| OLS | −2.92 ± 1.92 |
| Ridge | −2.06 ± 1.23 |
| GBR | −1.51 ± 1.21 |

---

### Key Findings

1. **All models produce negative R² on temporally honest evaluation.** Macro features correlate contemporaneously but don't reliably predict out of sample.
2. **GBR overfits on random splits (R² = 0.99) but performs worst on temporal splits (R² = −0.28).** Tree ensembles memorize training-era patterns that invert during regime shifts.
3. **Autoregressive lags are the most useful features.** Adding 4-week and 13-week lags improved RMSE from 0.48 to 0.40.
4. **Feature importance dominated by Labor_Force_Participation (~58% Gini).** U6_Underemployment is the only macro feature contributing meaningful signal beyond autoregression.
5. **Enormous CV variance (±1.2 to ±1.9)** confirms the macro-to-earnings relationship is regime-dependent and unstable.

---

### Monte Carlo / Scenario Analysis (Streamlit dashboard)
- Gaussian copula simulation: fits marginals to training features, infers correlation matrix, draws correlated samples via Cholesky decomposition
- Ridge model used for predictions on simulated inputs
- Three preset scenarios: Tight labor market, Recession, Stagflation
- Reports mean, median, 95% CI, convergence diagnostics

---

### Dashboard Pages
1. **Overview** — Actual vs predicted time series, model comparison table, feature importance (with "predictive, not causal" caveat)
2. **What-if predictor** — Slider controls for all 12 features, live model predictions, z-score deviation chart
3. **Monte Carlo simulation** — Configurable iterations/distribution, forecast histogram with CI, convergence diagnostic
4. **Scenario comparison** — Side-by-side distributions for Tight/Recession/Stagflation scenarios