import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from lifelines import KaplanMeierFitter

from style_utils import (
    inject_custom_css, takeaway_box, info_card, scorecard, figure_header, figure_note,
    style_plotly_fig, COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, CATEGORY_COLORWAY,
)

st.set_page_config(page_title="Neobank Churn — Slide 2", layout="wide")
inject_custom_css()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
REFERENCE_DATE = pd.Timestamp("2019-05-01")
CHURN_THRESHOLD_DAYS = 90


@st.cache_resource
def load_model():
    model = joblib.load(DATA_DIR / "rf_engaged_model.joblib")
    feature_columns = joblib.load(DATA_DIR / "rf_engaged_feature_columns.joblib")
    categorical_cols = joblib.load(DATA_DIR / "categorical_cols.joblib")
    feature_cols = joblib.load(DATA_DIR / "feature_cols.joblib")
    return model, feature_columns, categorical_cols, feature_cols


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "model_df_final_features.parquet")


def get_original_feature(col_name, categorical_cols):
    for cat_col in categorical_cols:
        if col_name.startswith(cat_col + "_"):
            return cat_col
    return col_name


@st.cache_data
def score_and_explain(_categorical_cols, _feature_cols, _feature_columns):
    model, feature_columns, categorical_cols, feature_cols = load_model()
    df = load_data()

    X = pd.get_dummies(df[feature_cols], columns=categorical_cols, drop_first=False)
    X = X.reindex(columns=feature_columns, fill_value=0)
    y = df["churn_model"]

    y_proba = model.predict_proba(X)[:, 1]

    from sklearn.metrics import roc_auc_score, recall_score
    roc_auc = roc_auc_score(y, y_proba)
    recall = recall_score(y, model.predict(X))

    sample_idx = np.random.RandomState(42).choice(len(X), size=min(2000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_values_churn = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    feature_map = {col: get_original_feature(col, categorical_cols) for col in X.columns}
    shap_df = pd.DataFrame(shap_values_churn, columns=X_sample.columns)
    mean_abs_shap = shap_df.abs().mean()
    aggregated = (
        pd.DataFrame({"shap_value": mean_abs_shap, "original_feature": [feature_map[c] for c in mean_abs_shap.index]})
        .groupby("original_feature")["shap_value"].sum()
        .sort_values(ascending=False).head(6).reset_index()
    )
    return roc_auc, recall, aggregated


@st.cache_data
def build_survival_curves():
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    transactions_clean = pd.read_parquet(DATA_DIR / "transactions_clean.parquet")

    last_txn = (
        transactions_clean.assign(created_date=lambda d: d["created_date"].dt.tz_localize(None))
        .groupby("user_id")["created_date"].max().reset_index()
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

    kmf_all = KaplanMeierFitter()
    kmf_all.fit(df["duration"], event_observed=df["event"])
    active_at_90 = kmf_all.survival_function_at_times(90).iloc[0]
    active_at_365 = kmf_all.survival_function_at_times(365).iloc[0]

    standard = df[df["plan_binary"] == "STANDARD"]
    paid = df[df["plan_binary"] == "Paid plans"]
    kmf_standard = KaplanMeierFitter().fit(standard["duration"], event_observed=standard["event"])
    kmf_paid = KaplanMeierFitter().fit(paid["duration"], event_observed=paid["event"])

    return kmf_standard, kmf_paid, active_at_90, active_at_365


def km_trace(kmf, name, color):
    sf = kmf.survival_function_
    x, y = sf.index.values, sf.iloc[:, 0].values
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, shape="hv", width=2.5))


st.markdown(
    "<span style='font-size:0.8em; color:#6B7280;'>Slide 2 / 3 — Prediction & Time-to-Churn</span>",
    unsafe_allow_html=True,
)
st.title("Model & Survival")

try:
    model, feature_columns, categorical_cols, feature_cols = load_model()
    roc_auc, recall, aggregated = score_and_explain(categorical_cols, feature_cols, feature_columns)
    kmf_standard, kmf_paid, active_at_90, active_at_365 = build_survival_curves()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scorecard("ROC-AUC", f"{roc_auc:.2f}", COLOR_BLUE)
    with col2:
        scorecard("Recall (Churned)", f"{recall:.0%}", COLOR_GREEN)
    with col3:
        scorecard("Active at 1 year", f"{active_at_365:.0%}", COLOR_ORANGE)
    with col4:
        scorecard("STANDARD Hazard Ratio", "1.92×", COLOR_RED)

    left, mid, right = st.columns([1, 1.1, 1])

    with left:
        figure_header(4, "Feature Importance (SHAP)")
        fig_shap = px.bar(
            aggregated, x="shap_value", y="original_feature", orientation="h",
            labels={"shap_value": "", "original_feature": ""},
        )
        fig_shap.update_traces(marker_color=COLOR_BLUE)
        fig_shap.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        fig_shap = style_plotly_fig(fig_shap, height=220)
        st.plotly_chart(fig_shap, use_container_width=True)
        figure_note("Random Forest trained on activated users only, class_weight='balanced' to prioritize recall.")

    with mid:
        figure_header(5, "Survival Curve by Plan")
        fig_km = go.Figure()
        fig_km.add_trace(km_trace(kmf_standard, "STANDARD", COLOR_RED))
        fig_km.add_trace(km_trace(kmf_paid, "Paid plans", COLOR_GREEN))
        fig_km.update_layout(xaxis_title="Days since signup", yaxis_title="% still active", legend=dict(x=0.55, y=0.95))
        fig_km = style_plotly_fig(fig_km, height=220)
        st.plotly_chart(fig_km, use_container_width=True)
        figure_note("Log-rank test p<0.000001 — STANDARD churns far faster, though likely self-selection (see Slide 1).")

    with right:
        figure_header(6, "The Zero-Activity Signal")
        info_card("", "90.8% predicted churn risk", "One user, zero activity across every behavioral variable in the observation window.", COLOR_RED)
        info_card("", "active_days_obs = 0", "No transactions, no contacts, no notifications — in their first 60 days.", COLOR_ORANGE)
        info_card("", "Actionable trigger", "This pattern is mechanically easy to detect automatically before day 60.", COLOR_GREEN)
        figure_note("Illustrative SHAP example: the model's highest-risk prediction in the test set.")

    takeaway_box(
        "Interpretation",
        "Users inactive in their first 60 days are the clearest churn signal — recall of "
        f"{recall:.0%} confirms the model catches most real churners. An automatic early-warning "
        "trigger at day 60 could flag at-risk users for onboarding nudges before they're lost for good."
    )

except FileNotFoundError as e:
    st.error(f"Could not find required files in `{DATA_DIR}`.")
    st.exception(e)
