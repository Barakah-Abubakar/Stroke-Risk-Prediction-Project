import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline model
model = joblib.load("stroke_risk_model.pkl")

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

st.title("🧠 Stroke Risk Percentage Predictor")
st.write("Estimate stroke risk based on simple health indicators.")

# User Inputs
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=45,
    step=1
    
)

gender = st.selectbox("Gender", ["Male", "Female"])

high_blood_pressure = st.selectbox(
    "High Blood Pressure", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

irregular_heartbeat = st.selectbox(
    "Irregular Heartbeat", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

snoring_sleep_apnea = st.selectbox(
    "Snoring / Sleep Apnea", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

chest_pain = st.selectbox(
    "Chest Pain", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Prediction
if st.button("Predict Stroke Risk"):
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "high_blood_pressure": high_blood_pressure,
        "irregular_heartbeat": irregular_heartbeat,
        "snoring_sleep_apnea": snoring_sleep_apnea,
        "chest_pain": chest_pain
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"🧠 Estimated Stroke Risk: **{prediction:.2f}%**")
