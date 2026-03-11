import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()



import streamlit as st
import numpy as np
import keras
import os
import re
from datetime import date

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# Username Validation
# -----------------------------
def valid_username(name):
    parts = name.strip().split()
    return len(parts) == 2

# -----------------------------
# Password Validation
# -----------------------------
def valid_password(password):
    pattern = r"^(?=(?:.*\d){2})(?=(?:.*[A-Za-z]){1})[A-Za-z\d]{3}$"
    return re.match(pattern, password)

# -----------------------------
# Login Page
# -----------------------------
def login():

    st.title("🔐 Doctor Login")

    st.write("Username must contain **two names**.")
    st.write("Password must contain **two numbers and one letter** (example: 12A).")

    fullname = st.text_input("Doctor Full Name")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if not valid_username(fullname):
            st.error("Username must contain exactly two names")
            return

        if not valid_password(password):
            st.error("Password must contain exactly two numbers and one letter")
            return

        st.session_state.logged_in = True
        st.session_state.doctor = fullname

        st.success("Login successful")
        st.rerun()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_diabetes_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_full_model.keras")
    return keras.models.load_model(model_path)

# -----------------------------
# Diabetes Prediction Page
# -----------------------------
def diabetes_app():

    model = load_diabetes_model()

    st.title("🩺 Diabetes Health Predictor")

    st.write(f"Logged in as **Dr. {st.session_state.doctor}**")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    # -----------------------------
    # Patient Details
    # -----------------------------
    st.subheader("Patient Information")

    patient_name = st.text_input("Patient Full Name")
    patient_id = st.text_input("Patient Unique Number")
    visit_date = st.date_input("Visit Date", value=date.today())

    st.divider()

    st.subheader("Health Indicators")

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

    with st.form("health_form"):

        cols = st.columns(2)
        user_inputs = []

        for i, name in enumerate(feature_names):

            with cols[i % 2]:

                if name == "BMI":
                    val = st.number_input(
                        "BMI",
                        min_value=10.0,
                        max_value=70.0,
                        value=25.0,
                        step=0.1
                    )

                elif name == "Age":
                    val = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=30
                    )

                elif name == "GenHlth":
                    val = st.slider(
                        "General Health",
                        1,
                        5,
                        3,
                        help="1 = Excellent, 5 = Poor"
                    )

                elif name == "PhysHlth":
                    val = st.slider(
                        "Physical Health (Bad Days)",
                        0,
                        30,
                        0,
                        help="Days physical health was not good in past 30 days"
                    )

                else:
                    val = st.selectbox(
                        name,
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )

                user_inputs.append(val)

        submit = st.form_submit_button("Predict Diabetes Risk")

    # -----------------------------
    # Prediction
    # -----------------------------
    if submit:

        if patient_name == "" or patient_id == "":
            st.error("Please fill in patient details first")
            return

        data = np.array([user_inputs], dtype="float32")

        prediction = model.predict(data, verbose=0)

        risk_score = float(prediction[0][0])
        patient_data = {
    "doctor": st.session_state.doctor,
    "patient_name": patient_name,
    "patient_id": patient_id,
    "visit_date": str(visit_date),
    "BMI": user_inputs[0],
    "Age": user_inputs[1],
    "GenHlth": user_inputs[2],
    "PhysHlth": user_inputs[3],
    "HighBP": user_inputs[4],
    "HighChol": user_inputs[5],
    "PhysActivity": user_inputs[6],
    "HeartDiseaseorAttack": user_inputs[7],
    "DiffWalk": user_inputs[8],
    "Smoker": user_inputs[9],
    "risk_score": risk_score
}

db.collection("patients").add(patient_data)


        st.divider()

        st.subheader("Prediction Result")

        st.write("Patient Name:", patient_name)
        st.write("Patient ID:", patient_id)
        st.write("Visit Date:", visit_date)

        if risk_score > 0.5:
            st.error(f"⚠️ High Diabetes Risk ({risk_score*100:.1f}%)")
        else:
            st.success(f"✅ Low Diabetes Risk ({risk_score*100:.1f}%)")

        st.caption("This system assists doctors and is not a medical diagnosis.")

# -----------------------------
# App Control
# -----------------------------
if st.session_state.logged_in:
    diabetes_app()
else:
    login()
