import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("Stroke Risk Percentage Predictor")


# Load and train model

@st.cache_resource
def train_model():
    df = pd.read_csv("stroke_risk_dataset_v2.csv")

    features = [
        "age",
        "gender",
        "high_blood_pressure",
        "irregular_heartbeat",
        "snoring_sleep_apnea",
        "chest_pain"
    ]

    target = "stroke_risk_percentage"

    X = df.drop("stroke_risk_percentage", axis=1)
    y = df["stroke_risk_percentage"]

    categorical_features = ["gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             categorical_features)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)
    return model


model = train_model()


# User Inputs


age = st.number_input("Age", min_value=18, max_value=100, value=45)
gender = st.selectbox("Gender", ["Male", "Female"])

high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
irregular_heartbeat = st.selectbox("Irregular Heartbeat", ["No", "Yes"])
snoring_sleep_apnea = st.selectbox("Snoring / Sleep Apnea", ["No", "Yes"])
chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

# Convert Yes/No to 0/1
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "high_blood_pressure": 1 if high_blood_pressure == "Yes" else 0,
    "irregular_heartbeat": 1 if irregular_heartbeat == "Yes" else 0,
    "snoring_sleep_apnea": 1 if snoring_sleep_apnea == "Yes" else 0,
    "chest_pain": 1 if chest_pain == "Yes" else 0
}])




# Prediction

if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Stroke Risk: **{prediction:.2f}%**")
