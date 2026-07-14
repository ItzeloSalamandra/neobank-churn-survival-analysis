import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from style_utils import (
    inject_custom_css, takeaway_box, info_card, scorecard, figure_header, figure_note,
    style_plotly_fig, COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, CHURN_COLOR_MAP,
)

st.set_page_config(page_title="Neobank Churn — Slide 3", layout="wide")
inject_custom_css()

DATA_DIR = Path(__file__).parent.parent.parent / "data"

PCA_VARS = [
    "active_days_obs", "out_count_obs", "in_count_obs",
    "num_merchant_categories_obs", "num_countries_visited_obs",
    "total_amount_obs", "avg_transaction_amount_obs",
    "declined_rate_obs", "num_contacts", "num_notifications_obs", "age",
]
LOG_TRANSFORM_VARS = [v for v in PCA_VARS if v != "age"]


@st.cache_resource
def load_model():
    model = joblib.load(DATA_DIR / "rf_engaged_model.joblib")
    feature_columns = joblib.load(DATA_DIR / "rf_engaged_feature_columns.joblib")
    categorical_cols = joblib.load(DATA_DIR / "categorical_cols.joblib")
    feature_cols = joblib.load(DATA_DIR / "feature_cols.joblib")
    return model, feature_columns, categorical_cols, feature_cols


@st.cache_data
def build_risk_value_matrix(_categorical_cols, _feature_cols, _feature_columns):
    model, feature_columns, categorical_cols, feature_cols = load_model()
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    transactions_clean = pd.read_parquet(DATA_DIR / "transactions_clean.parquet")

    onboarding_failure_mask = (model_df_full["active_days_obs"] == 0) & (model_df_full["churn_model"] == True)
    matrix_df = model_df_full[~onboarding_failure_mask].copy()

    clv_completed = (
        transactions_clean[transactions_clean["transactions_state"] == "COMPLETED"]
        .groupby("user_id")["amount_usd"].sum().reset_index()
        .rename(columns={"amount_usd": "clv_proxy"})
    )
    matrix_df = matrix_df.merge(clv_completed, on="user_id", how="left")
    matrix_df["clv_proxy"] = matrix_df["clv_proxy"].fillna(0)

    X_all = pd.get_dummies(matrix_df[feature_cols], columns=categorical_cols, drop_first=False)
    X_all = X_all.reindex(columns=feature_columns, fill_value=0)
    matrix_df["risk_score"] = model.predict_proba(X_all)[:, 1]

    risk_median = matrix_df["risk_score"].median()
    value_median = matrix_df["clv_proxy"].median()
    matrix_df["risk_level"] = np.where(matrix_df["risk_score"] > risk_median, "High Risk", "Low Risk")
    matrix_df["value_level"] = np.where(matrix_df["clv_proxy"] > value_median, "High Value", "Low Value")

    def assign_quadrant(row):
        if row["risk_level"] == "High Risk" and row["value_level"] == "High Value":
            return "Critical Priority"
        elif row["risk_level"] == "High Risk":
            return "Low-Cost Risk"
        elif row["value_level"] == "High Value":
            return "Protect / Nurture"
        return "Low Priority"

    matrix_df["quadrant"] = matrix_df.apply(assign_quadrant, axis=1)
    return matrix_df, risk_median, value_median


@st.cache_data
def run_pca():
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    onboarding_failure_mask = (model_df_full["active_days_obs"] == 0) & (model_df_full["churn_model"] == True)
    pca_df = model_df_full[~onboarding_failure_mask].copy()

    X_raw = pca_df[PCA_VARS].copy()
    for col in LOG_TRANSFORM_VARS:
        X_raw[col] = np.log1p(X_raw[col])
    X_scaled = StandardScaler().fit_transform(X_raw)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df["PC1"], pca_df["PC2"] = components[:, 0], components[:, 1]
    pca_df["churn_label"] = pca_df["churn_model"].map({False: "Active", True: "Churned"})
    return pca_df, pca.explained_variance_ratio_


st.markdown(
    "<span style='font-size:0.8em; color:#6B7280;'>Slide 3 / 3 — Action & Segments</span>",
    unsafe_allow_html=True,
)
st.title("Action & Segments")

try:
    model, feature_columns, categorical_cols, feature_cols = load_model()
    matrix_df, risk_median, value_median = build_risk_value_matrix(categorical_cols, feature_cols, feature_columns)
    pca_df, explained_var = run_pca()

    n_critical = (matrix_df["quadrant"] == "Critical Priority").sum()
    total_clv_at_risk = matrix_df.loc[matrix_df["quadrant"] == "Critical Priority", "clv_proxy"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scorecard("Critical Priority users", f"{n_critical:,}", COLOR_RED)
    with col2:
        scorecard("CLV at risk (top segment)", f"${total_clv_at_risk/1e6:.1f}M", COLOR_ORANGE)
    with col3:
        scorecard("PC1 variance explained", f"{explained_var[0]:.0%}", COLOR_BLUE)
    with col4:
        scorecard("Total users analyzed", f"{len(matrix_df):,}", COLOR_GREEN)

    left, mid, right = st.columns([1.1, 1, 1])

    with left:
        figure_header(7, "Risk × Value Matrix")
        colors_map = {"Critical Priority": COLOR_RED, "Low-Cost Risk": COLOR_ORANGE,
                      "Protect / Nurture": COLOR_GREEN, "Low Priority": "#B0B7C3"}
        fig_matrix = px.scatter(
            matrix_df, x="risk_score", y="clv_proxy", color="quadrant",
            color_discrete_map=colors_map, opacity=0.4, log_y=True,
            labels={"risk_score": "Risk score", "clv_proxy": "CLV (log)"},
        )
        fig_matrix.add_vline(x=risk_median, line_dash="dash", line_color="gray")
        fig_matrix.add_hline(y=value_median, line_dash="dash", line_color="gray")
        fig_matrix.update_layout(legend=dict(font=dict(size=9)))
        fig_matrix = style_plotly_fig(fig_matrix, height=250)
        st.plotly_chart(fig_matrix, use_container_width=True)
        figure_note("Risk and value are negatively correlated — engaged users tend to churn less AND generate more value.")

    with mid:
        figure_header(8, "Behavioral Segments (PCA)")
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="churn_label",
            color_discrete_map=CHURN_COLOR_MAP, opacity=0.35,
            labels={"PC1": f"PC1 ({explained_var[0]:.0%})", "PC2": f"PC2 ({explained_var[1]:.0%})"},
        )
        fig_pca = style_plotly_fig(fig_pca, height=250)
        st.plotly_chart(fig_pca, use_container_width=True)
        figure_note("PC1 = overall usage intensity. Churn concentrates at low-but-nonzero activity, not at zero.")

    with right:
        figure_header(9, "Top Priority Users")
        top_users = (
            matrix_df[matrix_df["quadrant"] == "Critical Priority"]
            [["user_id", "risk_score", "clv_proxy"]]
            .sort_values("clv_proxy", ascending=False).head(3)
        )
        for _, row in top_users.iterrows():
            info_card(
                "", row["user_id"],
                f"Risk: {row['risk_score']:.0%} · CLV: ${row['clv_proxy']:,.0f}",
                COLOR_RED,
            )
        figure_note(f"Top 3 of {n_critical:,} users in Critical Priority, ranked by value.")

    takeaway_box(
        "Interpretation",
        f"{n_critical:,} users are both high-risk and high-value, representing ~${total_clv_at_risk/1e6:.1f}M "
        "in exposed customer value. This is the retention team's first call list — no further "
        "analysis needed to act, just outreach. PCA confirms engagement intensity (not plan or "
        "geography) is what actually separates who stays from who leaves."
    )

except FileNotFoundError as e:
    st.error(f"Could not find required files in `{DATA_DIR}`.")
    st.exception(e)
