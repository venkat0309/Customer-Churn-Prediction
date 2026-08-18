"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit web dashboard for Customer Churn Prediction.

Provides:
  1. Interactive Customer Risk Calculator / Scoring form
  2. Batch CSV Upload and prediction exporter
  3. Exploratory Data Analysis & Visualizations tab
  4. Model Performance Metrics & Confusion Matrices tab
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import add_domain_features
from src.predict import score_customers
from src.utils import load_config

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark glassmorphic styling
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_saved_model():
    model_path = Path("models/xgboost_pipeline.pkl")
    if not model_path.exists():
        st.error("Model file models/xgboost_pipeline.pkl not found. Please train models first.")
        st.stop()
    return joblib.load(model_path)


pipeline = load_saved_model()

# ── Title Header ─────────────────────────────────────────────────────────────
st.title("🔮 Telecom Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn risks in real-time, analyze key retention drivers, and score batch data using Random Forest and XGBoost machine learning models.")

tabs = st.tabs([
    "🎯 Single Customer Predictor",
    "📁 Batch Prediction (CSV Upload)",
    "📊 EDA & Visualizations",
    "📈 Model Performance & Metrics"
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Single Customer Predictor
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Interactive Churn Risk Calculator")
    st.write("Enter customer subscription details below to get an instant churn risk assessment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Demographic Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.markdown("### 📡 Services & Usage")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("### 💳 Account & Billing")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
        num_products = st.slider("Number of Subscribed Products", min_value=1, max_value=9, value=3)
        support_calls = st.slider("Support Calls Received", min_value=0, max_value=10, value=2)
        avg_monthly_gb_download = st.slider("Avg Monthly Download (GB)", min_value=0.0, max_value=300.0, value=45.0, step=5.0)

    if st.button("🚀 Calculate Churn Probability"):
        input_data = pd.DataFrame([{
            "customer_id": "CUST-SINGLE",
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "support_calls": support_calls,
            "avg_monthly_gb_download": avg_monthly_gb_download,
        }])

        scored = score_customers(pipeline, input_data, threshold=0.569)
        prob = scored.iloc[0]["churn_probability"]
        segment = scored.iloc[0]["risk_segment"]
        pred = scored.iloc[0]["churn_prediction"]

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
        with res_col2:
            st.metric("Prediction", "CHURN" if pred == 1 else "RETAIN")
        with res_col3:
            st.metric("Risk Segment", segment)

        if segment == "High Risk":
            st.error("⚠️ **High Risk Customer**: Immediate intervention recommended. Consider offering a contract upgrade discount or complimentary tech support/security package.")
        elif segment == "Medium Risk":
            st.warning("⚠️ **Medium Risk Customer**: Monitor engagement and promote loyalty rewards.")
        else:
            st.success("✅ **Low Risk Customer**: Normal operational retention flow.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Batch CSV Upload
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Batch CSV Scoring & Prediction Exporter")
    st.write("Upload a raw customer CSV file to score thousands of customer records simultaneously.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        st.write(f"Uploaded dataset contains **{len(raw_batch):,} rows** and **{len(raw_batch.columns)} columns**.")
        st.dataframe(raw_batch.head(5))

        if st.button("Score Uploaded Customers"):
            scored_batch = score_customers(pipeline, raw_batch, threshold=0.569)
            st.success(f"Scored {len(scored_batch):,} customer records successfully!")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("High Risk Customers", f"{(scored_batch['risk_segment'] == 'High Risk').sum():,}")
            col_b.metric("Medium Risk Customers", f"{(scored_batch['risk_segment'] == 'Medium Risk').sum():,}")
            col_c.metric("Low Risk Customers", f"{(scored_batch['risk_segment'] == 'Low Risk').sum():,}")

            st.dataframe(scored_batch.head(20))

            csv_data = scored_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored CSV Predictions",
                data=csv_data,
                file_name="scored_churn_predictions.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: EDA & Visualizations
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Exploratory Data Analysis & Visualizations")

    eda_dir = Path("reports/figures/eda")
    if eda_dir.exists():
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(str(eda_dir / "01_churn_distribution.png"), caption="Overall Churn Distribution", use_column_width=True)
            st.image(str(eda_dir / "03_categorical_churn_rates.png"), caption="Churn Rate by Categorical Feature", use_column_width=True)
        with img_col2:
            st.image(str(eda_dir / "02_numerical_distributions.png"), caption="Numerical Distributions by Churn", use_column_width=True)
            st.image(str(eda_dir / "04_correlation_heatmap.png"), caption="Feature Correlation Heatmap", use_column_width=True)
    else:
        st.info("Run `python main.py` locally to generate all EDA visual figures.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Model Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Model Performance & SHAP Feature Importance")

    mod_dir = Path("reports/figures/models")
    if mod_dir.exists():
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.image(str(mod_dir / "model_comparison.png"), caption="Model Metric Comparison", use_column_width=True)
            st.image(str(mod_dir / "roc_curves.png"), caption="ROC Curves (AUC = 0.9426)", use_column_width=True)
            st.image(str(mod_dir / "xgb_confusion_matrix.png"), caption="XGBoost Confusion Matrix", use_column_width=True)
        with m_col2:
            st.image(str(mod_dir / "xgb_feature_importance.png"), caption="XGBoost Feature Importance", use_column_width=True)
            st.image(str(mod_dir / "pr_curves.png"), caption="Precision-Recall Curves", use_column_width=True)
            st.image(str(mod_dir / "threshold_analysis.png"), caption="Decision Threshold Optimization", use_column_width=True)
