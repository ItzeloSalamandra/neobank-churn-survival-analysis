import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "model_df_full.parquet")


st.title("🧹 Data Cleaning")

st.markdown("""
Before any analysis, I diagnosed and corrected the 4 source tables
(`users`, `devices`, `notifications`, `transactions`) from scratch, documenting
every cleaning decision with its justification.
""")


st.divider()
st.subheader("A note on date ranges")

st.info(
    "**users** signups run through **January 3, 2019**, while "
    "**transactions** activity continues through **May 16, 2019** — "
    "about 4.5 months where no new users sign up, but existing users "
    "keep transacting. This isn't a data quality issue: `REFERENCE_DATE` "
    "(May 1, 2019) sits inside this window on purpose, closing the "
    "analysis at a full month to avoid partial-month bias. The "
    "observation/outcome window scheme (60 days observation + minimum "
    "60 days outcome) already excludes the 62 users who signed up too "
    "late to have a reliable outcome by that date — so late cohorts "
    "aren't penalized, they're just correctly excluded when there isn't "
    "enough follow-up time to trust their label."
)


st.divider()
st.subheader("Key findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔧 Schema fix in `devices`")
    st.markdown("""
    The table arrived with generic columns (`string_field_0`, `string_field_1`)
    with no real names — typical of a CSV loaded without a header. By inspecting
    a data sample, I confirmed they represented `device_brand` and `user_id`.

    I also found that **the original file's header row had been loaded as a
    data record** (`user_id == "user_id"` literally) — I filtered it out before
    using the table.
    """)

with col2:
    st.markdown("#### 💰 Currency conversion bug")
    st.markdown("""
    I found transactions with absurd amounts (up to **$85 billion**), almost
    entirely confined to the `DECLINED` state. By comparing against the real
    maximum of `COMPLETED` transactions (~$612,766), I confirmed that any
    amount above **$1,000,000** was, unambiguously, part of the same
    conversion bug — not real economic activity.
    """)

st.divider()
st.subheader("Separating onboarding failure from real churn")

st.markdown("""
I found that **662 users (3.4% of the base)** never had any transactional
activity in their entire available history — they didn't "churn," they never
activated the product to begin with. I excluded them from the churn analysis
(which measures attrition of people who did use the product) and treat them
as a separate business segment, requiring a product intervention (improving
onboarding), not a traditional retention campaign.
""")

try:
    df = load_data()

    onboarding_failure_mask = (df["active_days_obs"] == 0) & (df["churn_model"] == True)
    n_onboarding_failures = onboarding_failure_mask.sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total users", f"{len(df):,}")
    col2.metric("Onboarding failures", f"{n_onboarding_failures:,}",
                f"{n_onboarding_failures/len(df):.1%}")
    col3.metric("Activated users (analyzed)", f"{len(df) - n_onboarding_failures:,}")

    st.divider()
    st.subheader("Active days distribution in the observation window")

    fig = px.histogram(
        df[~onboarding_failure_mask],
        x="active_days_obs",
        nbins=60,
        title="Active days in the first 60 days since signup (activated users)",
        labels={"active_days_obs": "Active days"},
    )
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Could not find data files in `{DATA_DIR}`.")
    st.exception(e)
