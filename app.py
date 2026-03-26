# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource  # caches the model for faster reloads
def load_classifier():
    """
    Load the updated Keras 3 model without deprecated batch_shape issues.
    Make sure the model was saved using Keras 3:
        model.save("diabetes_classifier_new.keras")
    """
    model = load_model("diabetes_classifier_new.keras", compile=False)
    return model

classifier = load_classifier()

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Diabetes Risk Prediction")

# -----------------------------
# USER INPUTS
# -----------------------------
st.sidebar.header("Input Patient Data")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("Patient Data")
st.write(input_df)

# Convert dataframe to numpy array for the model
input_array = input_df.to_numpy()

prediction = classifier.predict(input_array)
risk_percentage = prediction[0][0] * 100  # assuming output is a single sigmoid neuron

# -----------------------------
# DISPLAY RESULT
# -----------------------------
st.subheader("Diabetes Risk Prediction")
if risk_percentage < 20:
    st.success(f"✅ Low Diabetes Risk ({risk_percentage:.2f}%)")
elif risk_percentage < 50:
    st.warning(f"⚠️ Moderate Diabetes Risk ({risk_percentage:.2f}%)")
else:
    st.error(f"❌ High Diabetes Risk ({risk_percentage:.2f}%)")
