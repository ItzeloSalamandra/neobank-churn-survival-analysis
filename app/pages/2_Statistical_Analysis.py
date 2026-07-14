import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"

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
    return pd.read_parquet(DATA_DIR / "model_df_final_features.parquet")


def cramers_v(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    v = np.sqrt((chi2 / n) / min_dim)
    return chi2, p, v


def rank_biserial(group_a, group_b):
    u_stat, p = mannwhitneyu(group_a, group_b, alternative="two-sided")
    n1, n2 = len(group_a), len(group_b)
    r = 1 - (2 * u_stat) / (n1 * n2)
    return u_stat, p, r


@st.cache_data
def compute_effect_sizes(df):
    results = []
    for col in CATEGORICAL_VARS:
        table = pd.crosstab(df[col], df["churn_model"])
        chi2, p, v = cramers_v(table)
        results.append({"variable": col, "test": "Chi-square", "effect_size": v, "p_value": p})

    for col in CONTINUOUS_VARS:
        churned = df.loc[df["churn_model"], col].dropna()
        active = df.loc[~df["churn_model"], col].dropna()
        u_stat, p, r = rank_biserial(churned, active)
        results.append({"variable": col, "test": "Mann-Whitney U", "effect_size": abs(r), "p_value": p})

    return pd.DataFrame(results).sort_values("effect_size", ascending=False)


st.title("📈 Statistical Analysis")

st.markdown("""
I tested each variable's association with churn using **Chi-square + Cramér's V**
(categorical) and **Mann-Whitney U + rank-biserial correlation** (continuous).
I prioritize **effect size over p-value**: with ~19,000 users, even trivial
differences come out "statistically significant" — what matters is whether
the effect size is large enough to act on.
""")

try:
    df = load_data()
    effect_sizes = compute_effect_sizes(df)

    st.divider()
    st.subheader("Effect size ranking (largest to smallest)")

    fig = px.bar(
        effect_sizes,
        x="effect_size", y="variable", color="test",
        orientation="h",
        title="Effect size by variable (Cramér's V or |rank-biserial r|)",
        labels={"effect_size": "Effect size", "variable": ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cohen's conventions: 0.1 = small, 0.3 = medium, 0.5 = large. "
        "num_contacts, active_days_obs, and total_amount_obs show the strongest effects."
    )

    st.divider()
    st.subheader("A closer look at plan and country")

    col1, col2 = st.columns(2)

    with col1:
        plan_churn = df.groupby("plan")["churn_model"].mean().sort_values(ascending=False).reset_index()
        fig_plan = px.bar(
            plan_churn, x="plan", y="churn_model",
            title="Churn rate by plan",
            labels={"churn_model": "Churn rate", "plan": "Plan"},
        )
        fig_plan.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_plan, use_container_width=True)
        st.caption(
            "⚠️ Likely self-selection: already-committed users choose to pay, "
            "not that paying causes retention."
        )

    with col2:
        country_stats = df.groupby("country")["churn_model"].agg(["mean", "count"]).reset_index()
        country_reliable = country_stats[country_stats["count"] >= 200].sort_values("mean", ascending=False)
        fig_country = px.bar(
            country_reliable, x="country", y="mean",
            title="Churn rate by country (n ≥ 200)",
            labels={"mean": "Churn rate", "country": "Country"},
        )
        fig_country.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_country, use_container_width=True)

    with st.expander("View full effect size table"):
        st.dataframe(effect_sizes, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Could not find data files in `{DATA_DIR}`.")
    st.exception(e)
