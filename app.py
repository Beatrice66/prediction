import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os
import psycopg2
import joblib
import gdown
from tensorflow.keras.models import load_model

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
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
# LOAD FULL PIPELINE
# -----------------------------
@st.cache_resource
def load_pipeline():
    try:
        # File names
        scaler_path = "scaler.pkl"
        encoder_path = "encoder_model.keras"
        model_path = "diabetes_full_model.keras"

        # 👉 OPTIONAL: download if not present
        # Replace with your file IDs if needed
        # gdown.download("LINK", scaler_path)
        # gdown.download("LINK", encoder_path)
        # gdown.download("LINK", model_path)

        # Check files exist
        if not os.path.exists(scaler_path):
            st.error("Missing scaler.pkl")
            return None, None, None

        if not os.path.exists(encoder_path):
            st.error("Missing encoder_model.keras")
            return None, None, None

        if not os.path.exists(model_path):
            st.error("Missing diabetes_full_model.keras")
            return None, None, None

        # Load components
        scaler = joblib.load(scaler_path)
        encoder = load_model(encoder_path)
        model = load_model(model_path)

        return scaler, encoder, model

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None

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

    scaler, encoder, model = load_pipeline()

    if model is None:
        st.stop()

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

    # Inputs
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
                val = st.selectbox(feature, [0,1],
                                   format_func=lambda x: "Yes" if x==1 else "No")
            inputs.append(val)

    # Prediction
    if st.button("Predict Diabetes Risk"):

        if patient_name == "" or patient_id == "":
            st.warning("Please enter patient details")
            return

        try:
            data = np.array([inputs], dtype=float)

            # ✅ FULL PIPELINE
            data_scaled = scaler.transform(data)
            data_encoded = encoder.predict(data_scaled)
            prediction = model.predict(data_encoded)

            risk_score = float(prediction[0][0])

            st.divider()
            st.subheader("📊 Prediction Result")

            if risk_score > 0.5:
                st.error(f"⚠️ High Diabetes Risk ({risk_score*100:.2f}%)")
            else:
                st.success(f"✅ Low Diabetes Risk ({risk_score*100:.2f}%)")

            st.caption("⚠️ AI-assisted prediction — not a medical diagnosis")

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
