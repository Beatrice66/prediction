import streamlit as st
import numpy as np
import keras
import os

st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

st.title("🩺 Health Indicator Assessment")
st.write("Please provide the following health information for prediction.")

@st.cache_resource
def load_diabetes_model():
    # Loading the functional_16 model saved on 2026-03-05
    return keras.models.load_model('diabetes_full_model.keras')

model = load_diabetes_model()

# These match your 10 features in the correct order for the model input
feature_names = [
    "BMI", "Age", "GenHlth", "PhysHlth", "HighBP", 
    "HighChol", "PhysActivity", "HeartDiseaseorAttack", 
    "DiffWalk", "Smoker"
]

if model:
    with st.form("health_form"):
        cols = st.columns(2)
        user_inputs = []
        
        for i, name in enumerate(feature_names):
            with cols[i % 2]:
                # Using selectboxes for binary (0/1) features to make it user-friendly
                if name in ["HighBP", "HighChol", "PhysActivity", "HeartDiseaseorAttack", "DiffWalk", "Smoker"]:
                    val = st.selectbox(f"{name}", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                else:
                    # For numerical/scale features like BMI, Age, GenHlth, PhysHlth
                    val = st.number_input(f"{name}", value=0.0, step=1.0)
                
                user_inputs.append(val)
        
        submit = st.form_submit_button("Predict Diabetes Risk")

    if submit:
        # Reshape to (1, 10) as required by input_layer_2
        data = np.array([user_inputs], dtype="float32")
        prediction = model.predict(data)
        risk_score = float(prediction[0][0])
        
        st.divider()
        if risk_score > 0.5:
            st.error(f"### High Risk Detected")
            st.write(f"Probability: **{risk_score*100:.1f}%**")
        else:
            st.success(f"### Low Risk Detected")
            st.write(f"Probability: **{risk_score*100:.1f}%**")

        st.caption("Disclaimer: This model uses a neural network with a 0.3 dropout rate for estimation.")