import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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

    X = X = df.drop("stroke_risk_percentage", axis=1)
    y = df["stroke_risk_percentage"]

    categorical_features = ["gender"]
    numeric_features = [f for f in X if f != "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
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
high_bp = st.selectbox("High Blood Pressure", [0, 1])
irregular_heartbeat = st.selectbox("Irregular Heartbeat", [0, 1])
snoring = st.selectbox("Snoring / Sleep Apnea", [0, 1])
chest_pain = st.selectbox("Chest Pain", [0, 1])

input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "high_blood_pressure": high_bp,
    "irregular_heartbeat": irregular_heartbeat,
    "snoring_sleep_apnea": snoring,
    "chest_pain": chest_pain
}])


# Prediction

if st.button("Predict Stroke Risk"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"Estimated Stroke Risk: **{prediction:.2f}%**")
