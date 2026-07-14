import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Risk x Value Matrix", page_icon="🎯", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_resource
def load_model_and_config():
    model = joblib.load(DATA_DIR / "rf_engaged_model.joblib")
    feature_columns = joblib.load(DATA_DIR / "rf_engaged_feature_columns.joblib")
    categorical_cols = joblib.load(DATA_DIR / "categorical_cols.joblib")
    feature_cols = joblib.load(DATA_DIR / "feature_cols.joblib")
    return model, feature_columns, categorical_cols, feature_cols


@st.cache_data
def build_matrix_df(_categorical_cols, _feature_cols, _feature_columns):
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    transactions_clean = pd.read_parquet(DATA_DIR / "transactions_clean.parquet")

    onboarding_failure_mask = (model_df_full["active_days_obs"] == 0) & (model_df_full["churn_model"] == True)
    matrix_df = model_df_full[~onboarding_failure_mask].copy()

    clv_completed = (
        transactions_clean[transactions_clean["transactions_state"] == "COMPLETED"]
        .groupby("user_id")["amount_usd"]
        .sum()
        .reset_index()
        .rename(columns={"amount_usd": "clv_proxy"})
    )
    matrix_df = matrix_df.merge(clv_completed, on="user_id", how="left")
    matrix_df["clv_proxy"] = matrix_df["clv_proxy"].fillna(0)

    return matrix_df, _categorical_cols, _feature_cols, _feature_columns


@st.cache_data
def score_risk(matrix_df, _model, _categorical_cols, _feature_cols, _feature_columns):
    X_all = pd.get_dummies(matrix_df[_feature_cols], columns=_categorical_cols, drop_first=False)
    X_all = X_all.reindex(columns=_feature_columns, fill_value=0)
    matrix_df = matrix_df.copy()
    matrix_df["risk_score"] = _model.predict_proba(X_all)[:, 1]
    return matrix_df


def assign_quadrant(row):
    if row["risk_level"] == "High Risk" and row["value_level"] == "High Value":
        return "Critical Priority"
    elif row["risk_level"] == "High Risk" and row["value_level"] == "Low Value":
        return "Low-Cost Risk"
    elif row["risk_level"] == "Low Risk" and row["value_level"] == "High Value":
        return "Protect / Nurture"
    else:
        return "Low Priority"


st.title("🎯 Risk × Value Matrix")

st.markdown("""
Cross-referencing churn risk with customer value tells us **who to prioritize**
for retention outreach — high risk alone isn't enough, since a high-risk,
low-value user doesn't warrant the same effort as a high-risk, high-value one.
""")

try:
    model, feature_columns, categorical_cols, feature_cols = load_model_and_config()
    matrix_df, categorical_cols, feature_cols, feature_columns = build_matrix_df(
        categorical_cols, feature_cols, feature_columns
    )
    matrix_df = score_risk(matrix_df, model, categorical_cols, feature_cols, feature_columns)

    risk_median = matrix_df["risk_score"].median()
    value_median = matrix_df["clv_proxy"].median()

    matrix_df["risk_level"] = np.where(matrix_df["risk_score"] > risk_median, "High Risk", "Low Risk")
    matrix_df["value_level"] = np.where(matrix_df["clv_proxy"] > value_median, "High Value", "Low Value")
    matrix_df["quadrant"] = matrix_df.apply(assign_quadrant, axis=1)

    st.divider()
    st.subheader("Quadrant distribution")

    quadrant_counts = matrix_df["quadrant"].value_counts().reset_index()
    quadrant_counts.columns = ["quadrant", "count"]

    col1, col2 = st.columns([1, 2])
    with col1:
        for _, row in quadrant_counts.iterrows():
            st.metric(row["quadrant"], f"{row['count']:,}", f"{row['count']/len(matrix_df):.1%}")

    with col2:
        colors = {"Critical Priority": "#d62728", "Low-Cost Risk": "#ff7f0e",
                  "Protect / Nurture": "#1f4e9c", "Low Priority": "#7f7f7f"}
        fig = px.scatter(
            matrix_df, x="risk_score", y="clv_proxy", color="quadrant",
            color_discrete_map=colors, opacity=0.4,
            log_y=True,
            labels={"risk_score": "Risk Score (churn probability)", "clv_proxy": "CLV proxy (USD, log scale)"},
            title="Risk × Value Matrix",
        )
        fig.add_vline(x=risk_median, line_dash="dash", line_color="gray")
        fig.add_hline(y=value_median, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: risk and value are negatively correlated — highly engaged users tend to "
        "generate more value AND churn less, since the same product engagement drives both."
    )

    st.divider()
    st.subheader("Top priority users for retention outreach")

    priority_users = (
        matrix_df[matrix_df["quadrant"] == "Critical Priority"]
        [["user_id", "risk_score", "clv_proxy", "plan", "country", "num_contacts", "active_days_obs"]]
        .sort_values("clv_proxy", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(priority_users.head(20), use_container_width=True)
    st.caption(f"Showing top 20 of {len(priority_users):,} users in Critical Priority.")

except FileNotFoundError as e:
    st.error(f"Could not find required files in `{DATA_DIR}`.")
    st.exception(e)
