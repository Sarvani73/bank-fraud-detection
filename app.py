import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bank Fraud Detection", layout="centered")

st.title("💳 Bank Fraud Detection System")
st.markdown("This AI model predicts whether a transaction is **Fraudulent or Normal**.")

st.divider()

st.subheader("Enter Transaction Details")

input_data = []

# Time Feature
time = st.number_input("Transaction Time", value=0.0)
input_data.append(time)

# V1–V28 Features
for i in range(1, 29):
    value = st.number_input(f"Feature V{i}", value=0.0)
    input_data.append(value)

# Amount Feature
amount = st.number_input("Transaction Amount", value=0.0)
input_data.append(amount)

st.divider()

if st.button("🔍 Predict Transaction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"Fraud Probability: {probability:.4f}")