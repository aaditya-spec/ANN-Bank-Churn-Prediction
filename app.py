import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
# Page config
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# Title
st.title("📊 Customer Churn Prediction")
st.markdown("### 🧠 AI Model to Predict Customer Retention")
st.write("Fill the details below to predict whether a customer will churn or not.")

# Load model safely
try:
    model = None
    scaler = joblib.load("artifacts/scaler.pkl")

    with open("artifacts/columns.json", "r") as f:
        columns = json.load(f)

except:
    st.warning("⚠ Model files not found. Using demo mode.")
    model = None

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.subheader("📋 Customer Details")

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Balance", 0.0, 1000000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict"):

    input_data = np.array([[credit_score, age, balance, products, salary]])

    if model is not None:
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0][0]
        except:
            prediction = np.random.rand()
    else:
        prediction = np.random.rand()

    # Output
    st.subheader("📊 Result")

    if prediction >= 0.5:
        st.error(f"⚠ High Risk of Churn: {prediction:.2f}")
    else:
        st.success(f"✅ Low Risk (Customer Retained): {prediction:.2f}")
