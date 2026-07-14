import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

st.set_page_config(page_title="Random Forest Model", page_icon="🌲", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_resource
def load_model_and_config():
    model = joblib.load(DATA_DIR / "rf_engaged_model.joblib")
    feature_columns = joblib.load(DATA_DIR / "rf_engaged_feature_columns.joblib")
    categorical_cols = joblib.load(DATA_DIR / "categorical_cols.joblib")
    feature_cols = joblib.load(DATA_DIR / "feature_cols.joblib")
    return model, feature_columns, categorical_cols, feature_cols


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "model_df_final_features.parquet")


@st.cache_data
def prepare_engaged_split(df, _categorical_cols, _feature_cols, _feature_columns):
    onboarding_failure_mask = (df["active_days_obs"] == 0) & (df["churn_model"] == True)
    df_engaged = df[~onboarding_failure_mask].copy()

    X = pd.get_dummies(df_engaged[_feature_cols], columns=_categorical_cols, drop_first=False)
    X = X.reindex(columns=_feature_columns, fill_value=0)
    y = df_engaged["churn_model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


@st.cache_data
def compute_shap_importance(_model, X_test, _feature_cols_map):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    else:
        shap_values_churn = shap_values[:, :, 1]

    shap_df = pd.DataFrame(shap_values_churn, columns=X_test.columns)
    mean_abs_shap = shap_df.abs().mean()

    aggregated = (
        pd.DataFrame({
            "shap_value": mean_abs_shap,
            "original_feature": [_feature_cols_map.get(c, c) for c in mean_abs_shap.index],
        })
        .groupby("original_feature")["shap_value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return aggregated, shap_values_churn, explainer.expected_value


def get_original_feature(col_name, categorical_cols):
    for cat_col in categorical_cols:
        if col_name.startswith(cat_col + "_"):
            return cat_col
    return col_name


st.title("🌲 Random Forest Model")

st.markdown("""
I trained a Random Forest to predict churn among **activated users only**
(excluding the 662 onboarding-failure cases). I used `class_weight="balanced"`
to prioritize recall on churners over raw accuracy — for a retention use case,
missing a real churner is far more costly than flagging an active user by mistake.
""")

try:
    model, feature_columns, categorical_cols, feature_cols = load_model_and_config()
    df = load_data()
    X_test, y_test = prepare_engaged_split(df, categorical_cols, feature_cols, feature_columns)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.divider()
    st.subheader("Model performance")

    col1, col2, col3 = st.columns(3)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["Active", "Churned"])
    col1.metric("Recall (Churned)", f"{report['Churned']['recall']:.2f}")
    col2.metric("Precision (Churned)", f"{report['Churned']['precision']:.2f}")
    col3.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Active", "Churned"], y=["Active", "Churned"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.info(
        "⚠️ Accuracy alone (77%) is lower than a dummy baseline (86%) — that's intentional. "
        "The model trades accuracy for recall, since the business cost of missing a real "
        "churner is much higher than over-flagging an active user."
    )

    st.divider()
    st.subheader("Feature importance (SHAP, aggregated)")

    feature_cols_map = {col: get_original_feature(col, categorical_cols) for col in X_test.columns}
    aggregated_importance, shap_values_churn, expected_value = compute_shap_importance(
        model, X_test, feature_cols_map
    )

    fig_shap = px.bar(
        aggregated_importance, x="shap_value", y="original_feature",
        orientation="h",
        title="Mean |SHAP value| aggregated by original feature (one-hot columns combined)",
        labels={"shap_value": "Mean |SHAP value|", "original_feature": ""},
    )
    fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()
    st.subheader("Individual explainability: highest-risk user example")

    high_risk_idx = int(np.argmax(y_proba))
    st.markdown(f"**User at index {high_risk_idx} — predicted churn probability: {y_proba[high_risk_idx]:.1%}**")

    user_row = X_test.iloc[high_risk_idx]
    nonzero_features = user_row[user_row != 0]
    st.markdown(
        "This user shows **zero activity across all behavioral variables** during the "
        "observation window — the clearest possible case for the model: someone who signed "
        "up but never engaged with the product in a meaningful way."
    )
    st.dataframe(user_row.to_frame("value"), use_container_width=True)

except FileNotFoundError as e:
    st.error(
        f"Could not find required files in `{DATA_DIR}`. "
        "Make sure you ran the joblib.dump() cell at the end of 03_random_forest_model.ipynb."
    )
    st.exception(e)
