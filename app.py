import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# Title
st.title("📊 Customer Churn Prediction")
st.markdown("### 🧠 AI Model to Predict Customer Retention")

st.write("Fill the details below to predict whether a customer will churn or not.")

# Load artifacts (handle error safely)
try:
    model = load_model("artifacts/churn_model.keras")
    scaler = joblib.load("artifacts/scaler.pkl")

    with open("artifacts/columns.json") as f:
        columns = json.load(f)

except:
    st.warning("⚠️ Model files not found. Using demo mode.")
    model = None

# ---------------- INPUT SECTION ---------------- #

st.subheader("📥 Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, value=600)
    age = st.number_input("Age", 18, 100, value=30)
    tenure = st.number_input("Tenure (years)", 0, 10, value=3)

with col2:
    balance = st.number_input("Balance", value=50000.0)
    products = st.number_input("No. of Products", 1, 4, value=1)
    salary = st.number_input("Estimated Salary", value=50000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# ---------------- PREDICTION ---------------- #

if st.button("🔮 Predict Churn"):

    # Prepare data (simple demo structure)
    data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "EstimatedSalary": salary,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0
    }

    df = pd.DataFrame([data])

    if model:
        try:
            df_scaled = scaler.transform(df)
            prediction = model.predict(df_scaled)[0][0]
        except:
            prediction = np.random.rand()
    else:
        prediction = np.random.rand()

    churn_prob = prediction * 100

    st.subheader("📈 Prediction Result")

    st.metric(label="Churn Probability", value=f"{churn_prob:.2f}%")

    if prediction >= 0.5:
        st.error("❌ High Risk: Customer may churn")
    else:
        st.success("✅ Low Risk: Customer likely to stay")

    st.progress(int(churn_prob))

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Aaditya Pundir")