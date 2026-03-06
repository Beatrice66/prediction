import streamlit as st
import numpy as np
import keras
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    # 1. Load the Keras model (functional_16)
    # Ensure this file is in your C:\Users\ADMIN\OneDrive\Desktop\prediction\ folder
    model = keras.models.load_model('diabetes_full_model.keras')
    
    # 2. Load the Scaler
    # This scaler expects exactly 10 features as input
    scaler = joblib.load('scaler.pkl')
    
    return model, scaler

# Initial Check for files and modules
try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Ensure diabetes_full_model.keras, scaler.pkl, and scikit-learn are installed.")
    st.stop()

# --- App UI ---
st.title("🩺 Diabetes Health Indicator")
st.markdown("""
Enter the patient's health information below. This model uses 10 specific indicators 
to calculate the probability of diabetes risk.
""")

# The 10 features in the EXACT order your scaler and model expect
feature_names = [
    "BMI", 
    "Age", 
    "GenHlth", 
    "PhysHlth", 
    "HighBP", 
    "HighChol", 
    "PhysActivity", 
    "HeartDiseaseorAttack", 
    "DiffWalk", 
    "Smoker"
]

# Create the form
with st.form("input_form"):
    st.subheader("Patient Clinical Profile")
    cols = st.columns(2)
    user_inputs = []
    
    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            # Binary Categorical Inputs (0 or 1)
            if name in ["HighBP", "HighChol", "PhysActivity", "HeartDiseaseorAttack", "DiffWalk", "Smoker"]:
                val = st.selectbox(
                    f"{name}", 
                    options=[0, 1], 
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                    help=f"Select 1 for Yes and 0 for No regarding {name}"
                )
            # Numeric Inputs (Scales or raw numbers)
            else:
                if name == "BMI":
                    val = st.number_input(name, min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                elif name == "GenHlth":
                    val = st.slider(name, 1, 5, 3, help="1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor")
                elif name == "PhysHlth":
                    val = st.number_input(name, min_value=0.0, max_value=30.0, value=0.0, help="Days of poor physical health in the last 30 days")
                else: # Age
                    val = st.number_input(name, min_value=1.0, max_value=100.0, value=30.0)
            
            user_inputs.append(val)
    
    submit = st.form_submit_button("Generate Prediction")

# --- Prediction Logic ---
if submit:
    # 1. Convert inputs to a 2D numpy array (Shape: 1, 10)
    input_array = np.array([user_inputs], dtype="float32")
    
    try:
        # 2. Scale the data using the pre-loaded StandardScaler
        # This prevents the "X has 5 features, but StandardScaler is expecting 10" error
        input_scaled = scaler.transform(input_array)
        
        # 3. Pass scaled data to the model
        prediction = model.predict(input_scaled)
        risk_score = float(prediction[0][0])
        
        # Display results with visual feedback
        st.divider()
        if risk_score > 0.5:
            st.error(f"### Result: High Risk Detected")
            st.metric(label="Diabetes Risk Probability", value=f"{risk_score*100:.1f}%")
            st.warning("The patient shows indicators strongly associated with diabetes risk.")
        else:
            st.success(f"### Result: Low Risk Detected")
            st.metric(label="Diabetes Risk Probability", value=f"{risk_score*100:.1f}%")
            st.info("The patient's health indicators currently show a low risk profile.")

    except ValueError as ve:
        st.error(f"Feature Mismatch Error: {ve}")
        st.write(f"Current input count: {len(user_inputs)}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.caption("Disclaimer: This tool is for educational purposes and based on the BRFSS dataset.")