# Macroeconomic Forecasting of U.S. Wage Growth

**CS 3916 Machine Learning Final Project — Ian Solberg, April 2026**

### [Live Dashboard →](https://average-weekly-earnings-prediction-model-iasolb.streamlit.app/)

---

## Overview

This project investigates whether observable macroeconomic indicators can predict the year-over-year growth rate of U.S. average weekly earnings. Using 32 features drawn from the Federal Reserve Economic Data (FRED) API — spanning labor markets, inflation, monetary policy, financial conditions, housing, and output — we train and evaluate three regression models on weekly data from 2007 through 2026.

The central finding is that **macro indicators have strong contemporaneous correlation with wage growth but limited out-of-sample predictive power**. Under temporally honest evaluation (training on the past, testing on the future), all three models produce negative $R^{2}$, meaning they predict worse than a naive historical-mean baseline. This result is consistent across multiple temporal cross-validation windows and reflects the regime-dependent, non-stationary nature of the macro-to-earnings relationship.

The project includes a Streamlit dashboard with interactive model comparison, a what-if feature explorer, Monte Carlo simulation of forecast distributions, and macroeconomic scenario analysis powered by a custom Gaussian copula simulation framework.

---

## Repository structure

```
ML_Prediction_Project/
├── app.py                          # Streamlit dashboard (4 pages)
├── distro.py                       # Data loading helper for app + notebook
├── get_data.py                     # FRED data pipeline: pull, score, subset
├── FinalProjectNotebook-3916-IanSolberg.ipynb
│                                   # Main analysis notebook (Parts 1-7)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── fred_data.csv               # Full scored dataset (217+ columns)
│   ├── fred_subset.csv             # Research subset (34 columns)
│   └── models/
│       ├── ols.pkl                 # Trained LinearRegression
│       ├── ridge.pkl               # Trained Ridge (α=1.0)
│       └── gbr.pkl                 # Trained GradientBoostingRegressor
│
├── FRED_Loader/                    # Submodule: FRED API data pipeline
│   ├── load.py                     # Series fetcher + resampler
│   ├── series.py                   # 229 FRED series definitions
│   ├── macro_scores.py             # 56 derived scoring blocks
│   └── utils.py                    # Config dataclass
│
└── ResearchFramework/              # Submodule: Monte Carlo simulation engine
    ├── simulation.py               # Core engine (InputManager, copula draws,
    │                               #   scenarios, sensitivity, convergence)
    ├── plotter.py                  # Plotly visualization layer
    └── rh.py                       # ResearchHandler for variable specs
```

---

## Data pipeline

### Source

All data is sourced from the [FRED API](https://fred.stlouisfed.org/) via the `FRED_Loader` submodule. The loader pulls 229 economic time series spanning:

- **Inflation** (CPI components, PCE, PPI, breakevens, expectations)
- **GDP & output** (real/nominal GDP, industrial production, capacity utilization)
- **Labor market** (payrolls, unemployment, JOLTS, earnings, productivity)
- **Interest rates** (Fed funds, Treasury curve, TIPS, spreads)
- **Money & credit** (M2, bank lending, consumer credit, delinquencies)
- **Housing** (Case-Shiller, starts, permits, mortgage rates)
- **Consumption & income** (PCE, retail sales, personal income, savings rate)
- **International** (trade balance, exchange rates, USD indices)
- **Commodities** (oil, gas, metals, agriculture)
- **Fiscal** (federal debt, deficit, receipts, expenditures)

All series are resampled to **weekly frequency (Friday-aligned)** using last-observation-carried-forward for monthly/quarterly data and weekly means for daily data. The dataset spans **2000-01-07 through present**.

### Scoring layer

The `macro_scores.py` module applies 56 registered scoring blocks that produce derived columns in four categories:

**Derived series** — year-over-year rates, annualized short-window growth, real rates, labor market ratios:

$$\text{CPI YoY}\_{t} = \frac{\text{CPI}\_{t} - \text{CPI}\_{t-52}}{\text{CPI}\_{t-52}} \times 100$$

$$\text{Real FFR PCE}\_{t} = \text{FedFunds}\_{t} - \text{Core PCE YoY}\_{t}$$

$$\text{JOLTS UE Ratio}\_{t} = \frac{\text{Job Openings}\_{t}}{\text{Unemployment Rate}\_{t} \times \text{Labor Force} / 100}$$

**Regime flags** — binary indicators for yield curve inversion, Sahm rule trigger, high-yield stress, inflation de-anchoring:

$$\text{Flag Sahm}\_{t} = \mathbb{1}\left[\bar{U}\_{t}^{(13w)} - \min\_{s \in [t-52, t]} U\_{s} \geq 0.50\right]$$

**Continuous scores** — Taylor rule gap, mandate tension, housing pressure, activity momentum:

$$\text{Taylor}\_{t} = r^{*} + \pi\_{t} + 0.5(\pi\_{t} - \pi^{*}) + 0.5(u^{*} - u\_{t})$$

$$\text{Taylor Gap}\_{t} = \text{Taylor}\_{t} - \text{FFR}\_{t}$$

$$\text{Activity Momentum}\_{t} = \frac{1}{|S|}\sum\_{s \in S} \frac{x\_{s,t} - \bar{x}\_{s}^{(156w)}}{\sigma\_{s}^{(156w)}}$$

where $S = \lbrace\text{Industrial Production, Retail Sales, PCE, Payrolls}\rbrace$.

**Lead/lag signals** — shifted series for predictive analysis (e.g., M2 growth lagged 78 weeks against future inflation).

### Subsetting

`get_data.py` extracts a 34-column research subset from the full dataset, applies a `datetime.now()` trim to remove forward-filled stale observations beyond the last real FRED release, and computes the year-over-year dependent variable:

$$y\_{t} = \frac{E\_{t} - E\_{t-52}}{E\_{t-52}} \times 100$$

where $E\_{t}$ is Average Weekly Earnings (BLS series CES0500000011) at week $t$.

---

## Dependent variable

**Avg_Weekly_Earnings_YoY** — the year-over-year percent change in average weekly earnings for all private-sector employees.

The raw earnings series (`CES0500000011`) begins in March 2006 and is reported monthly, resampled to weekly via forward-fill. The YoY transformation:
- Removes the upward trend (non-stationarity), converting a bimodal level distribution into a roughly unimodal rate distribution
- Aligns the dependent with features already in rate-of-change form (CPI_YoY, Credit_Impulse)
- Produces ~940 valid weekly observations (March 2007 through present)
- Yields a series centered around 2-4% with a COVID-era spike to ~7%

---

## Feature set

The final model uses **12 features** selected from the original 32 via correlation analysis, with non-stationary level variables replaced by stationary scored alternatives:

| Feature | Source | Description |
|---------|--------|-------------|
| `Core_PCE_YoY` | Scored | Core PCE inflation, 52-week percent change |
| `JOLTS_UE_Ratio` | Scored | Job openings per unemployed person |
| `Personal_Savings_Rate` | FRED | Household savings as % of disposable income |
| `Labor_Force_Participation` | FRED | Civilian labor force participation rate |
| `Sahm_Indicator` | Scored | 3-month UE average minus 12-month UE minimum |
| `HY_OAS_Spread` | FRED | High-yield corporate bond OAS spread |
| `U6_Underemployment` | FRED | Broad underemployment rate |
| `Employment_Population_Ratio` | FRED | Employed persons / civilian population |
| `Payroll_Change_4w` | Scored | 4-week change in nonfarm payrolls |
| `Activity_Momentum` | Scored | Composite z-score of IP, retail, PCE, payrolls |
| `Earnings_YoY_Lag4` | Derived | Target variable lagged 4 weeks |
| `Earnings_YoY_Lag13` | Derived | Target variable lagged 13 weeks |

Three originally selected features (`Retail_Sales`, `JOLTS_Quits`, `Part_Time_Economic_Reasons`) were replaced with stationary alternatives after temporal evaluation revealed that trending level variables cause extrapolation failure when the test set is strictly in the future.

---

## Methodology

### Train/test split

Temporal split at 80/20: training on all observations up to July 2022, testing on July 2022 through April 2026 (~788 train, ~197 test). This ensures the model is evaluated on a genuinely future macro regime (the post-pandemic wage deceleration) that it has never seen during training.

### Models

**Model 1 — OLS Linear Regression (baseline)**

Ordinary least squares minimizes the sum of squared residuals with no regularization:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top} \mathbf{y}$$

$$\hat{y} = \mathbf{X}\hat{\boldsymbol{\beta}}$$

This provides a closed-form solution and serves as the interpretability baseline.

**Model 2 — Ridge Regression**

Ridge adds an $L\_{2}$ penalty to shrink coefficients toward zero, reducing variance at the cost of some bias:

$$\hat{\boldsymbol{\beta}}\_{\text{ridge}} = (\mathbf{X}^{\top} \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^{\top} \mathbf{y}$$

where $\alpha = 1.0$. The penalty discourages large coefficients, which helps when features are correlated. In practice, Ridge performed nearly identically to OLS, indicating multicollinearity was not a significant issue in the 12-feature set.

**Model 3 — Gradient Boosting Regressor**

An ensemble of sequential shallow decision trees where each tree $h\_{m}$ fits the residuals of the cumulative ensemble:

$$F\_{0}(\mathbf{x}) = \bar{y}$$

$$F\_{m}(\mathbf{x}) = F\_{m-1}(\mathbf{x}) + \eta \cdot h\_{m}(\mathbf{x})$$

where $\eta = 0.03$ is the learning rate, each tree has `max_depth=3`, and `n_estimators=200` with `subsample=0.8` for stochastic gradient boosting. The model captures nonlinear interactions and regime-dependent feature effects that linear models miss.

### Evaluation metrics

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum\_{i=1}^{n}(y\_{i} - \hat{y}\_{i})^{2}}$$

$$R^{2} = 1 - \frac{\sum\_{i=1}^{n}(y\_{i} - \hat{y}\_{i})^{2}}{\sum\_{i=1}^{n}(y\_{i} - \bar{y})^{2}}$$

Negative $R^{2}$ indicates the model predicts worse than simply guessing the training-set mean $\bar{y}$ for every observation.

### Cross-validation

`TimeSeriesSplit` with 5 folds ensures each fold trains on an expanding window of past data and tests on the next chronological segment. This prevents temporal leakage that inflates metrics in standard k-fold CV.

---

## Results

### Test set performance (temporal split, with lag features)

| Model | RMSE | MAE | $R^{2}$ |
|-------|------|-----|---------|
| OLS | 0.3977 | 0.3169 | 0.011 |
| Ridge ( $\alpha=1.0$ ) | 0.3965 | 0.3162 | 0.017 |
| GBR | 0.4525 | 0.3763 | -0.280 |

### TimeSeriesSplit CV (5-fold)

| Model | CV $R^{2}$ |
|-------|------------|
| OLS | $-2.92 \pm 1.92$ |
| Ridge | $-2.06 \pm 1.23$ |
| GBR | $-1.51 \pm 1.21$ |

### Key findings

1. **All models produce negative $R^{2}$ on temporally honest evaluation.** Macro features correlate with wage growth contemporaneously but do not reliably predict future wage growth out of sample.

2. **GBR overfits on random splits ( $R^{2} = 0.99$ ) but performs worst on temporal splits ( $R^{2} = -0.28$ ).** Tree-based models memorize training-era patterns that invert during regime shifts. Linear models extrapolate more gracefully.

3. **Autoregressive lags are the most useful features.** Adding 4-week and 13-week lagged earnings improved RMSE from 0.48 to 0.40. The best predictor of next quarter's wage growth is this quarter's wage growth.

4. **Feature importance is dominated by Labor_Force_Participation (~58% Gini importance), followed by the lag features.** U6_Underemployment is the only macro feature contributing meaningful signal beyond autoregression.

5. **Performance variance across CV folds is enormous** ( $\pm 1.2$ to $\pm 1.9$ ), confirming the macro-to-earnings relationship is regime-dependent and unstable.

---

## Monte Carlo simulation

The Streamlit dashboard includes a Monte Carlo forecast engine built on the `ResearchFramework` simulation module. The pipeline:

1. **Fit marginal distributions** to each of the 12 input features from training data (normal or empirical)
2. **Infer the empirical correlation matrix** from observed feature co-movements
3. **Draw correlated samples** via the Gaussian copula:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}\_{k}), \quad \mathbf{z}\_{c} = \mathbf{L}\mathbf{z}, \quad \mathbf{u} = \Phi(\mathbf{z}\_{c})$$

where $\mathbf{L}$ is the Cholesky decomposition of the correlation matrix and $\Phi$ is the standard normal CDF. Each $u\_{i} \in [0,1]$ is then transformed through the target marginal's inverse CDF:

$$x\_{i} = F\_{i}^{-1}(u\_{i})$$

4. **Run the trained Ridge model** on each simulated draw to produce a distribution of predicted wage growth outcomes
5. **Report** mean, median, 95% confidence interval, and convergence diagnostics

### Scenario comparison

Three preset macroeconomic scenarios override specific feature distributions while holding others at fitted values:

- **Tight labor market** — low U6, high JOLTS ratio, high employment-population ratio
- **Recession** — high U6, elevated Sahm indicator, wide HY spreads, low JOLTS ratio
- **Stagflation** — high Core PCE YoY, elevated U6, wide HY spreads

The `ScenarioComparator` runs each scenario through the same simulation pipeline and produces overlaid forecast distributions for comparison.

---

## Running locally

```bash
git clone --recurse-submodules https://github.com/iasolb/ML_Prediction_Project.git
cd ML_Prediction_Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the data pipeline (requires FRED API key in .env)
python get_data.py

# launch the dashboard
streamlit run app.py

# or open the notebook
jupyter notebook FinalProjectNotebook-3916-IanSolberg.ipynb
```

The `.env` file requires a FRED API key:
```
FRED_API_KEY=your_key_here
```

---

## Tech stack

- **Data**: FRED API via `fredapi`, pandas, numpy
- **Modeling**: scikit-learn (LinearRegression, Ridge, GradientBoostingRegressor)
- **Simulation**: custom Monte Carlo engine (Gaussian copula, Cholesky decomposition, scipy distributions)
- **Visualization**: Plotly (dashboard), matplotlib/seaborn (notebook)
- **Deployment**: Streamlit Cloud
- **Scoring**: custom macro_scores.py registry (56 blocks, 75+ derived columns)
