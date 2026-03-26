import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

st.title("Diabetes Risk Prediction")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    # load keras model safely
    model = load_model("diabetes_full_model.keras", compile=False)

    # load encoder + scaler
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, encoder, scaler


model, encoder, scaler = load_models()


# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
glucose = st.number_input("Glucose Level", 50, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 40, 200, 80)
insulin = st.number_input("Insulin", 0, 900, 80)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

pregnancies = st.number_input("Pregnancies", 0, 20, 1)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    try:

        # categorical dataframe
        cat_df = pd.DataFrame({
            "Gender":[gender],
            "Smoking":[smoking],
            "FamilyHistory":[family_history]
        })

        # encode categorical
        encoded = encoder.transform(cat_df)

        # numeric features
        num_features = np.array([[

            age,
            bmi,
            glucose,
            blood_pressure,
            insulin,
            skin_thickness,
            pregnancies

        ]])

        # scale numeric
        num_scaled = scaler.transform(num_features)

        # combine
        final_input = np.concatenate([num_scaled, encoded], axis=1)

        # ensure correct shape
        final_input = final_input.reshape(1,10)

        # prediction
        prediction = model.predict(final_input)

        prob = float(prediction[0][0])

        st.write("Prediction Probability:", prob)

        if prob > 0.5:
            st.error("High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
