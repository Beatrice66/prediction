import streamlit as st
import numpy as np
import keras
import os
from datetime import date
import firebase_admin
from firebase_admin import credentials, db
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

# -----------------------------
# Firebase Setup
# -----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-project-id-default-rtdb.firebaseio.com/'
    })

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "doctor" not in st.session_state:
    st.session_state.doctor = ""

# -----------------------------
# Bootstrap CSS
# -----------------------------
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# -----------------------------
# Validation
# -----------------------------
def valid_username(name):
    return len(name.strip().split()) == 2

def valid_password(password):
    # exactly 2 numbers + 1 letter, 3 characters total
    pattern = r"^(?=(?:.*\d){2})(?=(?:.*[A-Za-z]){1})[A-Za-z\d]{3}$"
    return re.match(pattern, password)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(os.path.join(os.path.dirname(__file__), "diabetes_full_model.keras"))

model = load_model()

# -----------------------------
# Firebase Save
# -----------------------------
def save_to_firebase(doctor, patient_name, patient_id, visit_date, risk):
    ref = db.reference("predictions")
    data = {
        "doctor": doctor,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "visit_date": str(visit_date),
        "risk_score": float(risk)
    }
    ref.push(data)

# -----------------------------
# Login Page
# -----------------------------
def login():
    st.markdown('<div class="card p-3 shadow-sm">', unsafe_allow_html=True)
    st.subheader("🔐 Doctor Login")
    st.write("Username must have **two names**, password **2 digits + 1 letter**")

    fullname = st.text_input("Full Name")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not valid_username(fullname):
            st.error("Username must be two names")
            return
        if not valid_password(password):
            st.error("Password must have 2 numbers + 1 letter")
            return
        st.session_state.logged_in = True
        st.session_state.doctor = fullname
        st.success("Login successful")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Page
# -----------------------------
def predictor_app():
    st.markdown('<div class="card p-3 shadow-sm">', unsafe_allow_html=True)
    st.subheader(f"🩺 Diabetes Predictor - Dr. {st.session_state.doctor}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("### Patient Details")
    patient_name = st.text_input("Patient Full Name")
    patient_id = st.text_input("Patient Unique ID")
    visit_date = st.date_input("Visit Date", date.today())

    st.markdown("### Health Indicators")
    feature_names = [
        "BMI","Age","GenHlth","PhysHlth",
        "HighBP","HighChol","PhysActivity",
        "HeartDiseaseorAttack","DiffWalk","Smoker"
    ]

    with st.form("health_form"):
        cols = st.columns(2)
        user_inputs = []
        for i, name in enumerate(feature_names):
            with cols[i%2]:
                if name == "BMI":
                    val = st.number_input("BMI", 10.0, 70.0, 25.0, step=0.1)
                elif name == "Age":
                    val = st.number_input("Age", 1, 120, 30)
                elif name == "GenHlth":
                    val = st.slider("General Health",1,5,3)
                elif name == "PhysHlth":
                    val = st.slider("Physical Health (Bad Days)",0,30,0)
                else:
                    val = st.selectbox(name,[0,1],format_func=lambda x:"Yes" if x==1 else "No")
                user_inputs.append(val)

        submit = st.form_submit_button("Predict")

    if submit:
        if patient_name=="" or patient_id=="":
            st.error("Fill patient details first")
            return
        data = np.array([user_inputs], dtype="float32")
        risk = float(model.predict(data, verbose=0)[0][0])

        st.markdown(f"### Prediction Result for {patient_name}")
        st.write(f"Patient ID: {patient_id}")
        st.write(f"Visit Date: {visit_date}")
        if risk>0.5:
            st.error(f"⚠️ High Risk ({risk*100:.1f}%)")
        else:
            st.success(f"✅ Low Risk ({risk*100:.1f}%)")

        save_to_firebase(st.session_state.doctor, patient_name, patient_id, visit_date, risk)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# App Control
# -----------------------------
if st.session_state.logged_in:
    predictor_app()
else:
    login()
