import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bank Fraud Detection", layout="centered")

st.title("💳 Bank Fraud Detection System")
st.markdown("""
This Machine Learning model predicts whether a credit card transaction is **Fraudulent or Legitimate**.

⚠️ Note: The dataset features (V1–V28) are PCA-transformed variables used to protect user privacy.
""")

st.divider()

st.subheader("Enter Basic Transaction Details")

# Basic Inputs
time = st.number_input("Transaction Time", value=0.0)
amount = st.number_input("Transaction Amount", value=0.0)

# Advanced PCA Features
with st.expander("🔍 Advanced Feature Inputs (PCA Components)"):
    st.write("These features are transformed numerical components used internally by the model.")
    pca_features = []
    for i in range(1, 29):
        value = st.number_input(f"Feature V{i}", value=0.0, key=f"V{i}")
        pca_features.append(value)

st.divider()

if st.button("🚀 Predict Transaction"):
    input_data = [time] + pca_features + [amount]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability Score: {probability:.4f}")