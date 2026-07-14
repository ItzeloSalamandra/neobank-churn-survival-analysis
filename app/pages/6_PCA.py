import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="PCA", page_icon="🔬", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"

PCA_VARS = [
    "active_days_obs", "out_count_obs", "in_count_obs",
    "num_merchant_categories_obs", "num_countries_visited_obs",
    "total_amount_obs", "avg_transaction_amount_obs",
    "declined_rate_obs", "num_contacts", "num_notifications_obs", "age",
]
LOG_TRANSFORM_VARS = [v for v in PCA_VARS if v != "age"]


@st.cache_data
def load_and_run_pca():
    model_df_full = pd.read_parquet(DATA_DIR / "model_df_full.parquet")

    onboarding_failure_mask = (model_df_full["active_days_obs"] == 0) & (model_df_full["churn_model"] == True)
    pca_df = model_df_full[~onboarding_failure_mask].copy()

    X_raw = pca_df[PCA_VARS].copy()
    X_transformed = X_raw.copy()
    for col in LOG_TRANSFORM_VARS:
        X_transformed[col] = np.log1p(X_transformed[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    pca_full = PCA()
    pca_full.fit(X_scaled)

    pca_3d = PCA(n_components=3)
    components_3d = pca_3d.fit_transform(X_scaled)

    pca_df["PC1"] = components_3d[:, 0]
    pca_df["PC2"] = components_3d[:, 1]
    pca_df["PC3"] = components_3d[:, 2]
    pca_df["churn_label"] = pca_df["churn_model"].map({False: "Active", True: "Churned"})

    loadings = pd.DataFrame(
        pca_3d.components_.T, columns=["PC1", "PC2", "PC3"], index=PCA_VARS,
    )

    return pca_df, pca_full.explained_variance_ratio_, loadings


st.title("🔬 PCA")

st.markdown("""
Random Forest excluded several activity variables (`out_count_obs`,
`num_merchant_categories_obs`, `in_count_obs`, `num_countries_visited_obs`) due
to high correlation with `active_days_obs` — that redundancy would have diluted
SHAP interpretability. PCA is designed exactly for this: combining correlated
variables into orthogonal components without losing the information they carry.
""")

try:
    pca_df, explained_var, loadings = load_and_run_pca()
    cumulative_var = np.cumsum(explained_var)

    st.divider()
    st.subheader("Explained variance")

    col1, col2, col3 = st.columns(3)
    col1.metric("PC1 alone", f"{explained_var[0]:.1%}")
    col2.metric("PC1 + PC2", f"{cumulative_var[1]:.1%}")
    col3.metric("Components for 80%", f"{np.argmax(cumulative_var >= 0.8) + 1}")

    scree_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained_var))],
        "cumulative_variance": cumulative_var,
    })
    fig_scree = px.line(
        scree_df, x="component", y="cumulative_variance", markers=True,
        title="Cumulative explained variance",
        labels={"cumulative_variance": "Cumulative variance explained", "component": "Number of components"},
    )
    fig_scree.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% variance")
    st.plotly_chart(fig_scree, use_container_width=True)

    st.divider()
    st.subheader("Users projected onto PC1 x PC2, colored by churn")

    fig_2d = px.scatter(
        pca_df, x="PC1", y="PC2", color="churn_label",
        color_discrete_map={"Active": "#1f4e9c", "Churned": "#d62728"},
        opacity=0.4,
        labels={"PC1": f"PC1 ({explained_var[0]:.1%})", "PC2": f"PC2 ({explained_var[1]:.1%})"},
        title="PC1 x PC2",
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    st.info(
        "Churned users don't occupy a distinct region of the PCA space — they're "
        "mixed throughout the cloud. This is consistent with the loadings: PC1/PC2 "
        "mostly capture transactional volume and diversity, not the variables that "
        "actually predict churn well (num_contacts, plan)."
    )

    with st.expander("Explore in 3D (drag to rotate)"):
        fig_3d = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3", color="churn_label",
            color_discrete_map={"Active": "#1f4e9c", "Churned": "#d62728"},
            opacity=0.6,
            title="PC1 x PC2 x PC3",
        )
        fig_3d.update_traces(marker=dict(size=3))
        st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()
    st.subheader("Component loadings")
    st.dataframe(loadings.sort_values("PC1", ascending=False), use_container_width=True)
    st.caption(
        "PC1 = general usage intensity (all activity-cluster variables load positively and evenly). "
        "PC2/PC3 mix age, transaction size, contacts, and notifications less cleanly."
    )

except FileNotFoundError as e:
    st.error(f"Could not find data files in `{DATA_DIR}`.")
    st.exception(e)
