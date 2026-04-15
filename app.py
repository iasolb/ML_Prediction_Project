"""
Wage Growth Forecasting Dashboard
===================================
Streamlit app for CS 3916 ML Final Project
Ian Solberg, April 2026
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from .distro import IMPORTANT_COLUMNS, get_data
from .ResearchFramework.simulation import (
    ConvergenceDiagnostics,
    InputManager,
    ModelFunction,
    MonteCarloEngine,
    Scenario,
    ScenarioComparator,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

TARGET = "Avg_Weekly_Earnings_YoY"
MODEL_DIR = "data/models"
FEATURES = IMPORTANT_COLUMNS

FEATURE_LABELS = {
    "Payroll_Change_4w": "Payroll change (4w)",
    "Activity_Momentum": "Activity momentum",
    "Core_PCE_YoY": "Core PCE YoY (%)",
    "JOLTS_UE_Ratio": "Job openings / unemployed",
    "Personal_Savings_Rate": "Personal savings rate (%)",
    "Labor_Force_Participation": "Labor force participation (%)",
    "Sahm_Indicator": "Sahm recession indicator",
    "HY_OAS_Spread": "High-yield OAS spread (pp)",
    "U6_Underemployment": "U6 underemployment (%)",
    "Employment_Population_Ratio": "Employment-population ratio (%)",
    "Earnings_YoY_Lag4": "Wage growth, 4-week lag (%)",
    "Earnings_YoY_Lag13": "Wage growth, 13-week lag (%)",
}

PALETTE = {
    "actual": "#2C2C2A",
    "ols": "#378ADD",
    "ridge": "#1D9E75",
    "gbr": "#D85A30",
    "split": "#E24B4A",
    "sim": "#7F77DD",
}


# ══════════════════════════════════════════════════════════════════════
# DATA + MODELS
# ══════════════════════════════════════════════════════════════════════


@st.cache_data
def load_full_df():
    """Load full CSV with dates and target, plus lag features."""
    df = pd.read_csv("data/fred_subset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["Earnings_YoY_Lag4"] = df[TARGET].shift(4)
    df["Earnings_YoY_Lag13"] = df[TARGET].shift(13)
    df = df.dropna(subset=[TARGET, "Earnings_YoY_Lag4", "Earnings_YoY_Lag13"])
    df = df.reset_index(drop=True)
    return df


@st.cache_data
def load_feature_data():
    """Use distro module for the feature matrix."""
    return get_data()


@st.cache_resource
def load_models():
    return {
        "OLS": joblib.load(f"{MODEL_DIR}/ols.pkl"),
        "Ridge": joblib.load(f"{MODEL_DIR}/ridge.pkl"),
        "GBR": joblib.load(f"{MODEL_DIR}/gbr.pkl"),
    }


# ══════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Wage Growth Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

full_df = load_full_df()
feature_df = load_feature_data()
models = load_models()

# temporal split
split_idx = int(len(full_df) * 0.8)
train_df = full_df.iloc[:split_idx]
test_df = full_df.iloc[split_idx:]
split_date = str(test_df["date"].iloc[0])

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]
y_train = train_df[TARGET]
y_test = test_df[TARGET]

# sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Overview", "What-if predictor", "Monte Carlo simulation", "Scenario comparison"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**CS 3916 ML Final Project**  \nIan Solberg, April 2026  \nData: FRED Economic Data"
)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Wage growth forecasting from macroeconomic indicators")
    st.markdown(
        "Predicting **year-over-year average weekly earnings growth** using "
        "labor market, inflation, financial, and output indicators from FRED. "
        "Models trained on 2007 to mid-2022, tested on mid-2022 to 2026."
    )

    predictions = {name: m.predict(X_test) for name, m in models.items()}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train samples", f"{len(X_train):,}")
    c2.metric("Test samples", f"{len(X_test):,}")
    best = min(
        predictions, key=lambda k: np.sqrt(mean_squared_error(y_test, predictions[k]))
    )
    c3.metric(
        "Best RMSE",
        f"{np.sqrt(mean_squared_error(y_test, predictions[best])):.3f}",
        delta=best,
    )
    c4.metric("Best R\u00b2", f"{r2_score(y_test, predictions[best]):.3f}")

    st.subheader("Actual vs predicted: temporal test set")
    toggles = st.multiselect(
        "Models to display", list(models.keys()), default=list(models.keys())
    )

    ctx = min(100, len(train_df))
    train_tail = train_df.iloc[-ctx:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_tail["date"],
            y=train_tail[TARGET],
            mode="lines",
            line=dict(color=PALETTE["actual"], width=1.5, dash="dot"),
            opacity=0.3,
            name="Train (context)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_df["date"],
            y=y_test,
            mode="lines",
            line=dict(color=PALETTE["actual"], width=2.5),
            name="Actual",
        )
    )

    model_colors = {
        "OLS": PALETTE["ols"],
        "Ridge": PALETTE["ridge"],
        "GBR": PALETTE["gbr"],
    }
    model_dashes = {"OLS": "solid", "Ridge": "solid", "GBR": "dash"}
    for name in toggles:
        r2 = r2_score(y_test, predictions[name])
        fig.add_trace(
            go.Scatter(
                x=test_df["date"],
                y=predictions[name],
                mode="lines",
                line=dict(color=model_colors[name], width=1.8, dash=model_dashes[name]),
                name=f"{name} (R\u00b2={r2:.3f})",
            )
        )

    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=PALETTE["split"], width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=split_date,
        y=1,
        yref="paper",
        text="Train / test split",
        showarrow=False,
        font=dict(color=PALETTE["split"], size=11),
        xanchor="left",
        yanchor="top",
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        yaxis_title="Avg weekly earnings YoY growth (%)",
        xaxis_title="Date",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model comparison")
    rows = []
    for name, preds in predictions.items():
        rows.append(
            {
                "Model": name,
                "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
                "MAE": round(mean_absolute_error(y_test, preds), 4),
                "R\u00b2": round(r2_score(y_test, preds), 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Feature importance (Gradient Boosting, Gini)")
    importances = pd.Series(
        models["GBR"].feature_importances_, index=FEATURES
    ).sort_values()
    fig_imp = go.Figure(
        go.Bar(
            x=importances.values,
            y=[FEATURE_LABELS.get(f, f) for f in importances.index],
            orientation="h",
            marker_color=PALETTE["sim"],
        )
    )
    fig_imp.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Feature importance (Gini)",
        margin=dict(l=200, t=20, b=40),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Predictive importance only. Does not imply causal effect.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2: WHAT-IF PREDICTOR
# ══════════════════════════════════════════════════════════════════════

elif page == "What-if predictor":
    st.title("Interactive what-if predictor")
    st.markdown("Adjust macro indicators and see how each model responds in real time.")

    feat_stats = X_train.describe().T
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown("#### Adjust features")
        user_inputs = {}
        for feat in FEATURES:
            lo = float(feat_stats.loc[feat, "min"])
            hi = float(feat_stats.loc[feat, "max"])
            med = float(feat_stats.loc[feat, "50%"])
            step = round((hi - lo) / 100, 4) or 0.01
            user_inputs[feat] = st.slider(
                FEATURE_LABELS.get(feat, feat), lo, hi, med, step, key=f"sl_{feat}"
            )

    with col_right:
        st.markdown("#### Model predictions")
        input_row = pd.DataFrame([user_inputs])
        for name, model in models.items():
            val = float(model.predict(input_row)[0])
            c = {
                "OLS": PALETTE["ols"],
                "Ridge": PALETTE["ridge"],
                "GBR": PALETTE["gbr"],
            }[name]
            st.markdown(
                f"<div style='padding:12px 16px;margin-bottom:8px;"
                f"border-left:4px solid {c};background:#f9f9f9;border-radius:4px;'>"
                f"<span style='font-size:14px;color:#666;'>{name}</span><br>"
                f"<span style='font-size:28px;font-weight:600;'>{val:.2f}%</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Feature deviations from training median")
        dev_data = []
        for feat in FEATURES:
            med = float(feat_stats.loc[feat, "50%"])
            std = float(feat_stats.loc[feat, "std"])
            z = (user_inputs[feat] - med) / std if std > 0 else 0
            dev_data.append(
                {"Feature": FEATURE_LABELS.get(feat, feat), "Z": round(z, 2)}
            )
        dev_df = pd.DataFrame(dev_data)
        fig_dev = go.Figure(
            go.Bar(
                x=dev_df["Z"],
                y=dev_df["Feature"],
                orientation="h",
                marker_color=[
                    PALETTE["gbr"] if z > 0 else PALETTE["ols"] for z in dev_df["Z"]
                ],
            )
        )
        fig_dev.update_layout(
            template="plotly_white",
            height=380,
            xaxis_title="Std devs from training median",
            margin=dict(l=200, t=10, b=40),
        )
        st.plotly_chart(fig_dev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3: MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════

elif page == "Monte Carlo simulation":
    st.title("Monte Carlo wage growth forecast")
    st.markdown(
        "Fits distributions to each input feature from training data, draws "
        "correlated samples, and runs them through the Ridge model."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        n_iter = st.select_slider(
            "Iterations", [1_000, 5_000, 10_000, 25_000, 50_000], 10_000
        )
    with mc2:
        dist_type = st.selectbox(
            "Distribution family", ["normal", "empirical"], index=0
        )

    @st.cache_data
    def run_simulation(n, dist, seed=42):
        train_features = get_data()
        mgr = InputManager()
        for col in FEATURES:
            try:
                mgr.fit_from_data(train_features, [col], dist_type=dist)
            except Exception:
                mgr.fit_from_data(train_features, [col], dist_type="empirical")
        if len(FEATURES) > 1:
            try:
                mgr.infer_correlation_from_data(train_features)
            except Exception:
                pass

        ridge = joblib.load(f"{MODEL_DIR}/ridge.pkl")
        model_fn = ModelFunction(
            lambda df: ridge.predict(df[FEATURES]), vectorized=True
        )
        engine = MonteCarloEngine(mgr, model_fn, n_iterations=n, seed=seed)
        result = engine.run()
        result.summarize()
        return result

    with st.spinner("Running simulation..."):
        result = run_simulation(n_iter, dist_type)

    s = result.summarize()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean", f"{s['mean']:.3f}%")
    c2.metric("Median", f"{s['median']:.3f}%")
    c3.metric("Std dev", f"{s['std']:.3f}")
    c4.metric("95% CI low", f"{s['ci_lower']:.3f}%")
    c5.metric("95% CI high", f"{s['ci_upper']:.3f}%")

    st.subheader("Forecast distribution")
    outcomes = np.array(result.outcomes).flatten()
    fig_h = go.Figure()
    fig_h.add_trace(
        go.Histogram(x=outcomes, nbinsx=80, marker_color=PALETTE["sim"], opacity=0.7)
    )
    fig_h.add_vline(
        x=s["mean"],
        line_color=PALETTE["actual"],
        line_width=2,
        annotation_text=f"Mean: {s['mean']:.2f}%",
        annotation_position="top right",
    )
    fig_h.add_vline(
        x=s["ci_lower"], line_dash="dash", line_color=PALETTE["split"], line_width=1.5
    )
    fig_h.add_vline(
        x=s["ci_upper"],
        line_dash="dash",
        line_color=PALETTE["split"],
        line_width=1.5,
        annotation_text="95% CI",
        annotation_position="top right",
        annotation_font_color=PALETTE["split"],
    )
    fig_h.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Predicted wage growth YoY (%)",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("Convergence diagnostic")
    conv_df = ConvergenceDiagnostics.running_statistics(outcomes)
    is_conv = ConvergenceDiagnostics.is_converged(outcomes)
    suggested_n = ConvergenceDiagnostics.suggest_n(outcomes)
    cc1, cc2 = st.columns(2)
    cc1.metric("Converged?", "Yes" if is_conv else "No")
    cc2.metric("Suggested N", f"{suggested_n:,}")

    fig_c = go.Figure()
    fig_c.add_trace(
        go.Scatter(
            x=conv_df["iteration"],
            y=conv_df["cumulative_mean"],
            mode="lines",
            line=dict(color=PALETTE["ridge"], width=2),
            name="Cumulative mean",
        )
    )
    se = 1.96 * conv_df["cumulative_std"] / np.sqrt(conv_df["iteration"])
    fig_c.add_trace(
        go.Scatter(
            x=conv_df["iteration"],
            y=conv_df["cumulative_mean"] + se,
            mode="lines",
            line=dict(color=PALETTE["ridge"], width=0.5, dash="dot"),
            showlegend=False,
        )
    )
    fig_c.add_trace(
        go.Scatter(
            x=conv_df["iteration"],
            y=conv_df["cumulative_mean"] - se,
            mode="lines",
            line=dict(color=PALETTE["ridge"], width=0.5, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(29,158,117,0.1)",
            showlegend=False,
        )
    )
    fig_c.update_layout(
        template="plotly_white",
        height=350,
        xaxis_title="Iteration",
        yaxis_title="Cumulative mean (%)",
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4: SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════

elif page == "Scenario comparison":
    st.title("Scenario comparison")
    st.markdown(
        "Compare predicted wage growth under different macroeconomic scenarios."
    )

    feat_stats = X_train.describe().T
    scenarios = [
        Scenario(
            "Tight labor market",
            overrides={
                "U6_Underemployment": {
                    "mean": float(feat_stats.loc["U6_Underemployment", "25%"]),
                    "std": 0.3,
                },
                "JOLTS_UE_Ratio": {
                    "mean": float(feat_stats.loc["JOLTS_UE_Ratio", "75%"]),
                    "std": 0.1,
                },
                "Employment_Population_Ratio": {
                    "mean": float(feat_stats.loc["Employment_Population_Ratio", "75%"]),
                    "std": 0.3,
                },
            },
        ),
        Scenario(
            "Recession",
            overrides={
                "U6_Underemployment": {
                    "mean": float(feat_stats.loc["U6_Underemployment", "max"]) * 0.9,
                    "std": 1.0,
                },
                "Sahm_Indicator": {"mean": 0.7, "std": 0.15},
                "HY_OAS_Spread": {
                    "mean": float(feat_stats.loc["HY_OAS_Spread", "75%"]) * 1.3,
                    "std": 1.0,
                },
                "JOLTS_UE_Ratio": {
                    "mean": float(feat_stats.loc["JOLTS_UE_Ratio", "25%"]) * 0.8,
                    "std": 0.1,
                },
            },
        ),
        Scenario(
            "Stagflation",
            overrides={
                "Core_PCE_YoY": {
                    "mean": float(feat_stats.loc["Core_PCE_YoY", "max"]) * 0.85,
                    "std": 0.5,
                },
                "U6_Underemployment": {
                    "mean": float(feat_stats.loc["U6_Underemployment", "75%"]),
                    "std": 0.8,
                },
                "HY_OAS_Spread": {
                    "mean": float(feat_stats.loc["HY_OAS_Spread", "75%"]),
                    "std": 0.8,
                },
            },
        ),
    ]

    n_sc = st.select_slider(
        "Iterations per scenario", [1_000, 5_000, 10_000, 25_000], 5_000, key="sc_n"
    )

    @st.cache_data
    def run_scenarios(n, seed=42):
        train_features = get_data()
        mgr = InputManager()
        for col in FEATURES:
            try:
                mgr.fit_from_data(train_features, [col], dist_type="normal")
            except Exception:
                mgr.fit_from_data(train_features, [col], dist_type="empirical")
        if len(FEATURES) > 1:
            try:
                mgr.infer_correlation_from_data(train_features)
            except Exception:
                pass

        ridge = joblib.load(f"{MODEL_DIR}/ridge.pkl")
        model_fn = ModelFunction(
            lambda df: ridge.predict(df[FEATURES]), vectorized=True
        )
        comp = ScenarioComparator(mgr, model_fn, scenarios, n_iterations=n, seed=seed)
        results = comp.run_all()
        summary = comp.compare_summary()
        return results, summary

    with st.spinner("Running scenarios..."):
        sc_results, sc_summary = run_scenarios(n_sc)

    st.subheader("Summary across scenarios")
    disp = sc_summary[
        ["scenario", "mean", "median", "std", "ci_lower", "ci_upper"]
    ].copy()
    disp.columns = [
        "Scenario",
        "Mean (%)",
        "Median (%)",
        "Std",
        "95% CI Low",
        "95% CI High",
    ]
    for col in disp.columns[1:]:
        disp[col] = disp[col].round(3)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("Forecast distributions by scenario")
    sc_colors = {
        "baseline": PALETTE["actual"],
        "Tight labor market": PALETTE["ridge"],
        "Recession": PALETTE["split"],
        "Stagflation": PALETTE["gbr"],
    }
    fig_sc = go.Figure()
    for name, res in sc_results.items():
        out = np.array(res.outcomes).flatten()
        fig_sc.add_trace(
            go.Histogram(
                x=out,
                nbinsx=60,
                name=name.capitalize() if name == "baseline" else name,
                marker_color=sc_colors.get(name, PALETTE["sim"]),
                opacity=0.55,
            )
        )
    fig_sc.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=450,
        xaxis_title="Predicted wage growth YoY (%)",
        yaxis_title="Count",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=30, b=60),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Scenario details")
    for sc in scenarios:
        with st.expander(sc.name):
            for var, ov in sc.overrides.items():
                st.markdown(
                    f"- **{FEATURE_LABELS.get(var, var)}**: {', '.join(f'{k}={v:.3f}' for k, v in ov.items())}"
                )
