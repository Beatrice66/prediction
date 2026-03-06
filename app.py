import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Title
st.title("Diabetes Risk Prediction App")

# Load model, scaler, encoder
@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("diabetes_full_model.keras")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_files()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading files: {e}")

# User Inputs
st.subheader("Enter Patient Information")

age = st.number_input("Age", 1, 120)
bmi = st.number_input("BMI", 10.0, 60.0)
glucose = st.number_input("Glucose Level", 50, 300)
blood_pressure = st.number_input("Blood Pressure", 40, 200)

# Prediction Button
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[age, bmi, glucose, blood_pressure]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")