# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# -----------------------------
# Model Definition (Keras 3)
# -----------------------------
def build_model():
    model = Sequential([
        Input(shape=(8,)),          # 8 features (adjust if your dataset has different)
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Load or create model
# -----------------------------
@st.cache_resource
def load_classifier():
    try:
        # Try to load existing Keras 3 model
        model = load_model("diabetes_classifier_new.keras", compile=False)
    except Exception:
        # If fails (e.g., first time), build a new one
        model = build_model()
        # Save it for future use
        model.save("diabetes_classifier_new.keras")
    return model

classifier = load_classifier()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Diabetes Prediction App (Keras 3 Compatible)")

st.write("Enter patient details:")

# Example features (adjust names as per your dataset)
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Input fields
inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs.append(value)

# Prediction button
if st.button("Predict Diabetes Risk"):
    data = np.array([inputs], dtype=np.float32)
    prediction = classifier.predict(data)[0][0]
    risk_percent = prediction * 100

    if risk_percent < 30:
        st.success(f"✅ Low Diabetes Risk ({risk_percent:.2f}%)")
    elif risk_percent < 70:
        st.warning(f"⚠️ Medium Diabetes Risk ({risk_percent:.2f}%)")
    else:
        st.error(f"❌ High Diabetes Risk ({risk_percent:.2f}%)")
