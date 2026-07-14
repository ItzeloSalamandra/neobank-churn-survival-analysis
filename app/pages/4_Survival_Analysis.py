import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

st.set_page_config(page_title="Survival Analysis", page_icon="⏳", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
REFERENCE_DATE = pd.Timestamp("2019-05-01")
CHURN_THRESHOLD_DAYS = 90


@st.cache_data
def build_survival_df():
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    transactions_clean = pd.read_parquet(DATA_DIR / "transactions_clean.parquet")

    last_txn = (
        transactions_clean
        .assign(created_date=lambda d: d["created_date"].dt.tz_localize(None))
        .groupby("user_id")["created_date"]
        .max()
        .reset_index()
        .rename(columns={"created_date": "last_transaction_date"})
    )

    df = model_df_full.merge(last_txn, on="user_id", how="left")

    onboarding_failure_mask = (df["active_days_obs"] == 0) & (df["churn_model"] == True)
    df = df[~onboarding_failure_mask].copy()

    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.tz_localize(None)
    df["days_since_last_txn"] = (REFERENCE_DATE - df["last_transaction_date"]).dt.days
    df["event"] = (df["days_since_last_txn"] > CHURN_THRESHOLD_DAYS).astype(int)
    df["duration"] = np.where(
        df["event"] == 1,
        (df["last_transaction_date"] - df["signup_date"]).dt.days,
        (REFERENCE_DATE - df["signup_date"]).dt.days,
    )
    df["plan_binary"] = np.where(df["plan"] == "STANDARD", "STANDARD", "Paid plans")
    return df


def km_to_plotly_trace(kmf, name, color):
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    x = sf.index.values
    y = sf.iloc[:, 0].values
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values

    trace = go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, shape="hv"))
    band = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor=color, opacity=0.15,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
    return [band, trace]


@st.cache_data
def fit_cox_model(survival_df):
    cox_df = survival_df[["duration", "event", "plan", "country",
                            "user_settings_crypto_unlocked", "num_contacts", "age"]].copy()
    cox_df["age_bin"] = pd.qcut(cox_df["age"], q=4, labels=False, duplicates="drop")
    cox_df["num_contacts_bin"] = pd.qcut(cox_df["num_contacts"], q=4, labels=False, duplicates="drop")
    cox_df = pd.get_dummies(cox_df, columns=["plan"], drop_first=True)

    strata_cols = ["country", "user_settings_crypto_unlocked", "age_bin", "num_contacts_bin"]
    covariate_cols = [c for c in cox_df.columns if c.startswith("plan_")]

    model_cols = ["duration", "event"] + covariate_cols + strata_cols
    cox_df = cox_df[model_cols].copy()
    cox_df[covariate_cols] = cox_df[covariate_cols].astype("float64")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", strata=strata_cols)
    return cph


st.title("⏳ Survival Analysis")

st.markdown("""
Instead of treating churn as a fixed yes/no outcome, survival analysis models
**when** a user is likely to churn, and correctly handles right-censoring —
users observed for different lengths of time (someone from January 2018 has
15 months of opportunity to "prove" they're still active; someone from
January 2019 only has the minimum 60-day window).
""")

try:
    survival_df = build_survival_df()

    st.divider()
    st.subheader("Overall survival curve")

    kmf_all = KaplanMeierFitter()
    kmf_all.fit(survival_df["duration"], event_observed=survival_df["event"])

    fig_all = go.Figure()
    for trace in km_to_plotly_trace(kmf_all, "All users", "#1f4e9c"):
        fig_all.add_trace(trace)
    fig_all.update_layout(
        title="Kaplan-Meier survival curve — all activated users",
        xaxis_title="Days since signup", yaxis_title="Probability of remaining active",
    )
    st.plotly_chart(fig_all, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Active at day 30", f"{kmf_all.survival_function_at_times(30).iloc[0]:.1%}")
    col2.metric("Active at day 90", f"{kmf_all.survival_function_at_times(90).iloc[0]:.1%}")
    col3.metric("Active at day 365", f"{kmf_all.survival_function_at_times(365).iloc[0]:.1%}")

    st.divider()
    st.subheader("Survival curves by plan type")

    standard_group = survival_df[survival_df["plan_binary"] == "STANDARD"]
    paid_group = survival_df[survival_df["plan_binary"] == "Paid plans"]

    kmf_standard = KaplanMeierFitter()
    kmf_standard.fit(standard_group["duration"], event_observed=standard_group["event"])

    kmf_paid = KaplanMeierFitter()
    kmf_paid.fit(paid_group["duration"], event_observed=paid_group["event"])

    fig_plan = go.Figure()
    for trace in km_to_plotly_trace(kmf_standard, "STANDARD", "#d62728"):
        fig_plan.add_trace(trace)
    for trace in km_to_plotly_trace(kmf_paid, "Paid plans", "#1f4e9c"):
        fig_plan.add_trace(trace)
    fig_plan.update_layout(
        title="Survival curves: STANDARD vs. Paid plans",
        xaxis_title="Days since signup", yaxis_title="Probability of remaining active",
    )
    st.plotly_chart(fig_plan, use_container_width=True)

    results = logrank_test(
        standard_group["duration"], paid_group["duration"],
        event_observed_A=standard_group["event"], event_observed_B=paid_group["event"],
    )
    st.metric("Log-rank test p-value", f"{results.p_value:.2e}")
    st.caption(
        "⚠️ Likely self-selection: already-committed users choose to pay, "
        "not that paying causes retention."
    )

    st.divider()
    st.subheader("Cox regression: hazard ratios (stratified model)")

    st.markdown("""
    I stratified by `country`, `user_settings_crypto_unlocked`, `age`, and
    `num_contacts` because they violated the proportional hazards assumption —
    stratifying lets each group have its own baseline hazard instead of forcing
    a constant effect over time. Only `plan` remains as a reportable covariate.
    """)

    cph = fit_cox_model(survival_df)
    hr_table = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
    hr_table.columns = ["Variable", "Hazard Ratio", "Lower 95%", "Upper 95%", "p-value"]

    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(
        x=hr_table["Hazard Ratio"], y=hr_table["Variable"],
        error_x=dict(
            type="data",
            symmetric=False,
            array=hr_table["Upper 95%"] - hr_table["Hazard Ratio"],
            arrayminus=hr_table["Hazard Ratio"] - hr_table["Lower 95%"],
        ),
        mode="markers", marker=dict(size=10),
    ))
    fig_hr.add_vline(x=1, line_dash="dash", line_color="gray")
    fig_hr.update_layout(
        title="Hazard ratios (plan, reference = METAL)",
        xaxis_title="Hazard Ratio (>1 = higher risk, <1 = protective)",
    )
    st.plotly_chart(fig_hr, use_container_width=True)

    st.dataframe(hr_table, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Could not find data files in `{DATA_DIR}`.")
    st.exception(e)
