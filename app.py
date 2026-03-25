import psycopg2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import os
import gdown

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = "final_diabetes_model.h5"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=13eu-b8zYlzwmFkC1N5lxXUBU3SJg8adI"
        gdown.download(url, model_path, quiet=False, fuzzy=True)

    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

# -----------------------------
# SAVE PREDICTION
# -----------------------------
def save_prediction(data):
    conn = get_connection()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO predictions(
            doctor_name, patient_name, patient_id,
            bmi, age, genhlth, physhlth,
            highbp, highchol, physactivity,
            heartdiseaseorattack, diffwalk, smoker,
            risk_score, prediction_date
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")

# -----------------------------
# FETCH HISTORY
# -----------------------------
def fetch_predictions():
    conn = get_connection()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_name, patient_name, patient_id, bmi, age, genhlth, physhlth,
                   highbp, highchol, physactivity, heartdiseaseorattack, diffwalk, smoker,
                   risk_score, prediction_date
            FROM predictions
            ORDER BY prediction_date DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return []

# -----------------------------
# LOAD MODEL FROM GOOGLE DRIVE
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = "final_diabetes_model.keras"

    # Download if not present
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1rdiFHg7thjaxKm4xY0d4kyFkdf48Kkux"
        gdown.download(url, model_path, quiet=False)

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_artifacts()
if model is None:
    st.stop()

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.title("🔐 Doctor Login")
    doctor_name = st.text_input("Doctor Full Name")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if doctor_name.strip() == "":
            st.warning("Please enter your name")
        else:
            st.session_state.logged_in = True
            st.session_state.doctor = doctor_name
            st.rerun()

# -----------------------------
# MAIN APP
# -----------------------------
def prediction_page():
    st.title("🩺 Diabetes Risk Predictor")
    st.write(f"👨‍⚕️ Logged in as: **{st.session_state.doctor}**")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    # Patient Info
    st.subheader("📋 Patient Information")
    patient_name = st.text_input("Patient Full Name")
    patient_id = st.text_input("Patient ID")
    current_date = datetime.now()

    st.write(f"Date: **{current_date.strftime('%Y-%m-%d')}**")
    st.divider()

    # Health Inputs
    st.subheader("🧾 Health Data")

    features = [
        "BMI","Age","GenHlth","PhysHlth","HighBP",
        "HighChol","PhysActivity","HeartDiseaseorAttack",
        "DiffWalk","Smoker"
    ]

    inputs = []
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            if feature == "BMI":
                val = st.number_input(feature, 10.0, 70.0, 25.0)
            elif feature == "Age":
                val = st.number_input(feature, 1, 120, 30)
            elif feature == "GenHlth":
                val = st.slider(feature, 1, 5, 3)
            elif feature == "PhysHlth":
                val = st.slider(feature, 0, 30, 0)
            else:
                val = st.selectbox(feature, [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            inputs.append(val)

    # Prediction
    if st.button("Predict Diabetes Risk"):

        if patient_name == "" or patient_id == "":
            st.warning("Please enter patient details")
            return

        try:
            data = np.array([inputs], dtype=float)

            prediction = model.predict(data)
            risk_score = float(prediction[0][0])

            st.divider()
            st.subheader("📊 Prediction Result")

            if risk_score > 0.5:
                st.error(f"⚠️ High Diabetes Risk ({risk_score*100:.2f}%)")
            else:
                st.success(f"✅ Low Diabetes Risk ({risk_score*100:.2f}%)")

            st.caption("⚠️ AI-assisted prediction — not a medical diagnosis")

            # Save to DB
            db_data = (
                st.session_state.doctor,
                patient_name,
                patient_id,
                *inputs,
                risk_score,
                current_date
            )

            save_prediction(db_data)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # History
    st.divider()
    st.subheader("📂 Prediction History")

    if st.button("Load Prediction History"):
        data = fetch_predictions()

        if data:
            df = pd.DataFrame(data, columns=[
                "Doctor","Patient","ID","BMI","Age","GenHlth",
                "PhysHlth","HighBP","HighChol","PhysActivity",
                "HeartDisease","DiffWalk","Smoker","Risk","Date"
            ])
            df["Risk"] = df["Risk"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(df)
        else:
            st.info("No predictions found.")

# -----------------------------
# APP CONTROL
# -----------------------------
if st.session_state.logged_in:
    prediction_page()
else:
    login_page()
