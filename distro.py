import pandas as pd

IMPORTANT_COLUMNS = [
    "Payroll_Change_4w",
    "Activity_Momentum",
    "Core_PCE_YoY",
    "JOLTS_UE_Ratio",
    "Personal_Savings_Rate",
    "Labor_Force_Participation",
    "Sahm_Indicator",
    "HY_OAS_Spread",
    "U6_Underemployment",
    "Employment_Population_Ratio",
    "Earnings_YoY_Lag4",
    "Earnings_YoY_Lag13",
]


def get_data(src: str = "data/fred_subset.csv") -> pd.DataFrame:
    df = pd.read_csv(src)
    df["Earnings_YoY_Lag4"] = df["Avg_Weekly_Earnings_YoY"].shift(4)
    df["Earnings_YoY_Lag13"] = df["Avg_Weekly_Earnings_YoY"].shift(13)
    df = df.dropna()
    df = pd.DataFrame(df[IMPORTANT_COLUMNS])
    return df
