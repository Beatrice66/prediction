import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model safely
@st.cache_resource
def load_diabetes_model():
    model = load_model("diabetes_full_model.keras")
    return model

model = load_diabetes_model()

st.title("🩺 Diabetes Risk Prediction App")

st.write("""
This application predicts the **risk of developing diabetes**
based on lifestyle and health information.
""")

st.header("Enter Patient Health Information")

# Numeric Inputs
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, step=0.1)
Age = st.number_input("Age", min_value=1, max_value=120, step=1)

# General Health Mapping
health_options = {
    "Excellent":1,
    "Very Good":2,
    "Good":3,
    "Fair":4,
    "Poor":5
}

GenHlth_label = st.selectbox(
    "General Health Status",
    list(health_options.keys())
)

GenHlth = health_options[GenHlth_label]

# Physical Health Days
PhysHlth = st.slider(
    "Number of Days Physical Health Was Not Good (Last 30 Days)",
    0, 30
)

# Yes/No Mapping
yes_no = {"No":0, "Yes":1}

HighBP = yes_no[st.selectbox("Do you have High Blood Pressure?", list(yes_no.keys()))]
HighChol = yes_no[st.selectbox("Do you have High Cholesterol?", list(yes_no.keys()))]
PhysActivity = yes_no[st.selectbox("Did you do Physical Activity in the Last 30 Days?", list(yes_no.keys()))]
HeartDisease = yes_no[st.selectbox("Do you have a history of Heart Disease or Heart Attack?", list(yes_no.keys()))]
DiffWalk = yes_no[st.selectbox("Do you have Difficulty Walking or Climbing Stairs?", list(yes_no.keys()))]
Smoker = yes_no[st.selectbox("Have you smoked at least 100 cigarettes in your lifetime?", list(yes_no.keys()))]

# Prediction
if st.button("Predict Diabetes Risk"):

    features = np.array([[BMI, Age, GenHlth, PhysHlth,
                          HighBP, HighChol, PhysActivity,
                          HeartDisease, DiffWalk, Smoker]])

    prob = model.predict(features)[0][0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Prediction Result")
    st.write("**Probability of Diabetes Risk:**", round(prob, 3))

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and not a medical diagnosis.")