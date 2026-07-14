import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Neobank Churn Analysis",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_data():
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")
    return model_df_full


st.title("📊 Neobank Churn Analysis")
st.markdown("### Portfolio project: churn prediction and user attrition analysis")

st.markdown("""
This dashboard summarizes a complete churn analysis for a neobank, covering:

- **Data cleaning**: diagnosis and correction of 4 source tables (users, devices, notifications, transactions)
- **Statistical analysis**: Chi-square and Mann-Whitney U with emphasis on effect size over p-value
- **Predictive model**: Random Forest with SHAP for explainability
- **Survival analysis**: Kaplan-Meier and Cox regression
- **Risk × Value matrix**: user prioritization for retention
- **PCA**: dimensionality reduction of user behavior

Use the sidebar to navigate between sections.
""")

try:
    df = load_data()

    st.divider()
    st.subheader("Quick data overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total users", f"{len(df):,}")
    col2.metric("Churn rate", f"{df['churn_model'].mean():.1%}")
    col3.metric("Countries", f"{df['country'].nunique()}")
    col4.metric("Distinct plans", f"{df['plan'].nunique()}")

except FileNotFoundError as e:
    st.error(
        f"Could not find data files in `{DATA_DIR}`. "
        "Run notebooks 01-06 first to generate the required Parquet files."
    )
    st.exception(e)
