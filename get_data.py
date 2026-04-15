import pandas as pd

from FRED_Loader.load import pull_fred
from FRED_Loader.series import ALL_SERIES
from FRED_Loader.utils import Config

FULL_SETUP: bool = False

FILENAME: str = "fred_data.csv"
OUTPUT_PATH: str = "data/"
START_DATE: str = "2000-01-01"
RESAMPLE_RULE: str = "W-FRI"
MEAN_FREQS: set[str] = {"D"}
APPLY_SCORES: bool = True
SERIES_OF_INTEREST = {**ALL_SERIES}

SUBSET_FILENAME: str = "fred_subset.csv"

# ── Dependent variable ────────────────────────────────────────────────
DEPENDENT: str = "Avg_Weekly_Earnings"

# ── Features to study ─────────────────────────────────────────────────
# fmt: off
COLUMNS_TO_STUDY: list[str] = [
    "date",
    DEPENDENT,

    # ── Labor Market Tightness ── (FRED) ──────────────────────────────
    "Unemployment_Rate",              # UNRATE
    "U6_Underemployment",             # U6RATE
    "Nonfarm_Payrolls",               # PAYEMS
    "JOLTS_Quits",                    # JTSQUL
    "Employment_Population_Ratio",    # EMRATIO
    "Part_Time_Economic_Reasons",     # LNS13025703
    "Labor_Force_Participation",      # CIVPART
    "Initial_Jobless_Claims",         # ICSA

    # ── Labor Market Tightness ── (scored) ────────────────────────────
    "JOLTS_UE_Ratio",                 # openings / unemployed (from JOLTS_Job_Openings, Unemployment_Rate)
    "Sahm_Indicator",                 # 3m UE avg − 12m UE min (from Unemployment_Rate)
    "Payroll_Change_4w",              # Δ payrolls 4-week (from Nonfarm_Payrolls)

    # ── Output & Demand ── (FRED) ─────────────────────────────────────
    "Industrial_Production",          # INDPRO
    "Capacity_Utilization",           # TCU
    "Retail_Sales",                   # RSAFS
    "Leading_Index_CB",               # USSLIND

    # ── Output & Demand ── (scored) ───────────────────────────────────
    "Activity_Momentum",              # composite z-score (from IP, Retail, PCE, Payrolls)

    # ── Inflation ── (scored) ─────────────────────────────────────────
    "CPI_YoY",                        # 52w pct_change (from Headline_CPI)
    "Core_PCE_YoY",                   # 52w pct_change (from Core_PCE)
    "Inflation_Momentum",             # 3m ann − YoY core PCE (from Core_PCE)

    # ── Monetary Policy ── (scored) ───────────────────────────────────
    "Real_FFR_PCE",                   # FFR − Core PCE YoY (from FedFunds_Rate, Core_PCE)
    "Taylor_Gap",                     # Taylor prescribed − FFR (from Core_PCE, UE, SEP_LR_FFR, FFR)

    # ── Financial Conditions ── (FRED) ────────────────────────────────
    "Treasury_10Y",                   # DGS10
    "HY_OAS_Spread",                  # BAMLH0A0HYM2
    "Chicago_Fed_Financial_Conditions",  # NFCI

    # ── Financial Conditions ── (scored) ──────────────────────────────
    "Flag_Curve_Inverted_10Y2Y",      # binary (from Yield_Curve_10Y_2Y)

    # ── Household Financial Health ── (FRED) ──────────────────────────
    "Personal_Savings_Rate",          # PSAVERT

    # ── Household Financial Health ── (scored) ────────────────────────
    "Credit_Impulse",                 # acceleration of credit growth (from Consumer_Credit_Total)

    # ── Housing ── (FRED) ─────────────────────────────────────────────
    "Mortgage_Rate_30Y",              # MORTGAGE30US

    # ── Housing ── (scored) ───────────────────────────────────────────
    "Housing_Pressure",               # price + rate deviation from trend (from Case_Shiller, Mortgage_Rate_30Y)

    # ── Demographics ── (FRED) ────────────────────────────────────────
    "Working_Age_Population",         # LFWA64TTUSM647S

    # ── Leading Signals ── (FRED) ─────────────────────────────────────
    # (Leading_Index_CB already listed above)

    # ── Leading Signals ── (scored) ───────────────────────────────────
    "Sentiment_Lag_13w",              # UMich sentiment lagged 13w (from UMich_Consumer_Sentiment)

    # ── Macro Regime ── (scored) ──────────────────────────────────────
    "Flag_Sahm_Triggered",            # binary recession flag (from Sahm_Indicator)
]
# fmt: on


def _transform(src: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS_TO_STUDY if c not in src.columns]
    if missing:
        print(f"\n⚠  Missing columns (will be NaN): {missing}")
    present = [c for c in COLUMNS_TO_STUDY if c in src.columns]
    return pd.DataFrame(src[present]).copy()


def get_research_subset(source: str) -> None:
    df = pd.read_csv(source)
    to_export = _transform(src=df)
    out_path = OUTPUT_PATH + SUBSET_FILENAME
    to_export.to_csv(path_or_buf=out_path, index=False)
    print(
        f"\nSubset → {out_path}  |  {len(to_export)} rows × {len(to_export.columns)} cols"
    )


def main():

    if FULL_SETUP:
        cfg: Config = Config(
            filename=FILENAME,
            output_path=OUTPUT_PATH,
            start=START_DATE,
            resample_rule=RESAMPLE_RULE,
            mean_freqs=MEAN_FREQS,
            series=SERIES_OF_INTEREST,
        )

        _ = pull_fred(config=cfg, apply_scores=APPLY_SCORES)
        source_path = OUTPUT_PATH + FILENAME
    else:
        source_path = OUTPUT_PATH + FILENAME

    get_research_subset(source=source_path)


if __name__ == "__main__":
    main()
