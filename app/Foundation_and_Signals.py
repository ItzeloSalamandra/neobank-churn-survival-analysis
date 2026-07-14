import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

from style_utils import (
    inject_custom_css, takeaway_box, info_card, scorecard, figure_header, figure_note,
    style_plotly_fig, COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, CATEGORY_COLORWAY,
)

st.set_page_config(page_title="Neobank Churn — Slide 1", layout="wide")
inject_custom_css()

DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORICAL_VARS = [
    "plan", "country", "device_brand",
    "attributes_notifications_marketing_push",
    "attributes_notifications_marketing_email",
]
CONTINUOUS_VARS = [
    "age", "num_contacts", "total_amount_obs", "avg_transaction_amount_obs",
    "active_days_obs", "declined_rate_obs", "num_notifications_obs",
]


@st.cache_data
def load_data():
    full_df = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    stats_df = pd.read_parquet(DATA_DIR / "model_df_final_features.parquet")
    return full_df, stats_df


def cramers_v(table):
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum().sum()
    min_dim = min(table.shape) - 1
    return chi2, p, np.sqrt((chi2 / n) / min_dim)


def rank_biserial(a, b):
    u_stat, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u_stat) / (len(a) * len(b))
    return u_stat, p, r


@st.cache_data
def compute_effect_sizes(df):
    results = []
    for col in CATEGORICAL_VARS:
        table = pd.crosstab(df[col], df["churn_model"])
        chi2, p, v = cramers_v(table)
        results.append({"variable": col, "effect_size": v})
    for col in CONTINUOUS_VARS:
        churned = df.loc[df["churn_model"], col].dropna()
        active = df.loc[~df["churn_model"], col].dropna()
        _, p, r = rank_biserial(churned, active)
        results.append({"variable": col, "effect_size": abs(r)})
    return pd.DataFrame(results).sort_values("effect_size", ascending=False).head(6)


st.markdown(
    "<span style='font-size:0.8em; color:#6B7280;'>Slide 1 / 3 — Foundation & Signals</span>",
    unsafe_allow_html=True,
)
st.title("Neobank Churn Analysis")

try:
    full_df, stats_df = load_data()
    onboarding_failure_mask = (full_df["active_days_obs"] == 0) & (full_df["churn_model"] == True)
    n_onboarding_failures = onboarding_failure_mask.sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scorecard("Total users", f"{len(full_df):,}", COLOR_BLUE)
    with col2:
        scorecard("Churn rate", f"{full_df.loc[~onboarding_failure_mask, 'churn_model'].mean():.1%}", COLOR_RED)
    with col3:
        scorecard("Countries", f"{full_df['country'].nunique()}", COLOR_GREEN)
    with col4:
        scorecard("Onboarding failures", f"{n_onboarding_failures:,}", COLOR_ORANGE)

    left, mid, right = st.columns([1.1, 0.9, 1.2])

    with left:
        figure_header(1, "Data Quality Highlights")
        info_card("", "Broken schema", "devices had no column names — recovered from sample values.", COLOR_BLUE)
        info_card("", "Currency bug", "DECLINED txns up to $85B — filtered above real completed max.", COLOR_ORANGE)
        info_card("", "687 never activated", "Excluded from churn — different problem (onboarding, not retention).", COLOR_RED)
        figure_note("Three issues found during raw-data diagnosis, corrected before any modeling.")

    with mid:
        figure_header(2, "Effect Size Ranking")
        effect_sizes = compute_effect_sizes(stats_df)
        fig = px.bar(
            effect_sizes, x="effect_size", y="variable", orientation="h",
            labels={"effect_size": "", "variable": ""},
        )
        fig.update_traces(marker_color=COLOR_BLUE)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        fig = style_plotly_fig(fig, height=220)
        st.plotly_chart(fig, use_container_width=True)
        figure_note("Effect size (Cramér's V / rank-biserial r) prioritized over p-value; n≈19,000 makes trivial gaps look 'significant.'")

    with right:
        figure_header(3, "Churn Rate by Plan")
        plan_churn = stats_df.groupby("plan")["churn_model"].mean().sort_values(ascending=False).reset_index()
        plan_churn["churn_model"] = plan_churn["churn_model"] * 100
        fig_plan = px.bar(
            plan_churn, x="plan", y="churn_model", color="plan",
            color_discrete_sequence=CATEGORY_COLORWAY,
            labels={"churn_model": "Churn rate (%)", "plan": ""},
            text_auto=".1f",
        )
        fig_plan.update_traces(textposition="outside", showlegend=False)
        fig_plan.update_layout(showlegend=False)
        fig_plan = style_plotly_fig(fig_plan, height=220)
        st.plotly_chart(fig_plan, use_container_width=True)
        figure_note("STANDARD stands out; paid plans show near-zero churn, likely self-selection rather than causation.")

    takeaway_box(
        "Interpretation",
        "Plan and contacts are the strongest signals, but likely reflect existing "
        "commitment, not a cause. Don't push plan upgrades — watch early activity instead (Slide 2)."
    )

except FileNotFoundError as e:
    st.error(f"Could not find data files in `{DATA_DIR}`.")
    st.exception(e)
