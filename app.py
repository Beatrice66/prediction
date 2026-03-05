import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="Diabetes Prediction Portal", layout="centered")

@st.cache_resource
def load_diabetes_model():
    # Ensure 'diabetes_full_model.keras' is in the same directory
    try:
        model = load_model('diabetes_full_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_diabetes_model()

st.title("🩺 Diabetes Risk Assessment")
st.markdown("""
This interface uses a deep learning model to predict the probability of diabetes 
based on 10 clinical features.
""")

# Creating a 2-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)
    f5 = st.number_input("Feature 5", value=0.0)

with col2:
    f6 = st.number_input("Feature 6", value=0.0)
    f7 = st.number_input("Feature 7", value=0.0)
    f8 = st.number_input("Feature 8", value=0.0)
    f9 = st.number_input("Feature 9", value=0.0)
    f10 = st.number_input("Feature 10", value=0.0)

st.divider()

if st.button("Analyze Risk", type="primary"):
    if model:
        # Prepare input data as per model's expected shape (batch_size, 10)
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]], dtype=np.float32)
        
        # Perform prediction
        prediction = model.predict(input_data)[0][0]
        
        # Display Results
        st.subheader("Results")
        risk_level = "High Risk" if prediction > 0.5 else "Low Risk"
        color = "red" if prediction > 0.5 else "green"
        
        st.markdown(f"### Probability: :{color}[{prediction:.2%}]")
        st.markdown(f"**Assessment:** {risk_level}")
        
        # Progress bar visual
        st.progress(float(prediction))
    else:
        st.error("Model not initialized. Please check the file path.")

st.sidebar.info("""
**Model Details:**
- **Architecture:** Functional + Sequential [cite: 1, 3]
- **Optimizer:** Adam (LR: 0.001) [cite: 5]
- **Output:** Sigmoid Activation 
""")