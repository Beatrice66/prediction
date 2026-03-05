import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("diabetes_full_model.keras")

st.title("Diabetes Risk Prediction App")

st.write("Enter the patient's health information below:")

# Input fields
BMI = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)

Age = st.number_input("Age", min_value=1, max_value=120, step=1)

GenHlth = st.selectbox(
    "General Health",
    [1,2,3,4,5],
    help="1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor"
)

PhysHlth = st.slider(
    "Number of Days Physical Health Was Not Good (Last 30 Days)",
    0, 30
)

HighBP = st.selectbox(
    "High Blood Pressure",
    [0,1],
    help="0 = No, 1 = Yes"
)

HighChol = st.selectbox(
    "High Cholesterol",
    [0,1],
    help="0 = No, 1 = Yes"
)

PhysActivity = st.selectbox(
    "Physical Activity in Last 30 Days",
    [0,1],
    help="0 = No, 1 = Yes"
)

HeartDisease = st.selectbox(
    "History of Heart Disease or Heart Attack",
    [0,1],
    help="0 = No, 1 = Yes"
)

DiffWalk = st.selectbox(
    "Difficulty Walking or Climbing Stairs",
    [0,1],
    help="0 = No, 1 = Yes"
)

Smoker = st.selectbox(
    "Smoker",
    [0,1],
    help="0 = No, 1 = Yes"
)

# Prediction button
if st.button("Predict Diabetes Risk"):

    features = np.array([[BMI, Age, GenHlth, PhysHlth,
                          HighBP, HighChol, PhysActivity,
                          HeartDisease, DiffWalk, Smoker]])

    prob = model.predict(features)[0][0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Prediction Result")

    st.write("**Probability of Diabetes Risk:**", round(prob, 3))

    if prediction == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")