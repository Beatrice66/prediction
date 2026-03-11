import streamlit as st
import numpy as np
import keras
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

st.title("🩺 Health Indicator Assessment")
st.write("Please provide the following health information for prediction:")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_diabetes_model():
    model_path = os.path.join(os.path.dirname(__file__), 'diabetes_full_model.keras')
    return keras.models.load_model(model_path)

model = load_diabetes_model()

# -----------------------------
# Feature Definitions
# -----------------------------
feature_names = [
    "BMI", "Age", "GenHlth", "PhysHlth", "HighBP", 
    "HighChol", "PhysActivity", "HeartDiseaseorAttack", 
    "DiffWalk", "Smoker"
]

# -----------------------------
# Form Inputs
# -----------------------------
with st.form("health_form"):
    cols = st.columns(2)
    user_inputs = []

    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            # Binary features: Yes/No selectboxes
            if name in ["HighBP", "HighChol", "PhysActivity", "HeartDiseaseorAttack", "DiffWalk", "Smoker"]:
                val = st.selectbox(
                    f"{name}", 
                    options=[0, 1], 
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
                )
            # Numerical features
            elif name == "BMI":
                val = st.number_input(name, min_value=10.0, max_value=70.0, value=25.0, step=0.1, help="Body Mass Index (kg/m²)")
            elif name == "Age":
                val = st.number_input(name, min_value=1, max_value=120, value=30, step=1, help="Age in years")
            # Health rating features as sliders
            elif name in ["GenHlth", "PhysHlth"]:
                val = st.slider(
                    name, min_value=0, max_value=5, value=3,
                    help=f"0 = Poor, 5 = Excellent"
                )
            user_inputs.append(val)

    submit = st.form_submit_button("Predict Diabetes Risk")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    data = np.array([user_inputs], dtype="float32")
    prediction = model.predict(data, verbose=0)
    risk_score = float(prediction[0][0])

    st.divider()
    if risk_score > 0.5:
        st.error(f"### High Risk Detected")
        st.write(f"Probability: **{risk_score*100:.1f}%**")
    else:
        st.success(f"### Low Risk Detected")
        st.write(f"Probability: **{risk_score*100:.1f}%**")

    st.caption("Disclaimer: This model uses a neural network with a 0.3 dropout rate for estimation.")

# -----------------------------
# Reset Form Button
# -----------------------------
if st.button("Reset Form"):
    st.experimental_rerun()
