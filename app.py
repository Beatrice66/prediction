import psycopg2
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_connection():
    # Connect to Supabase PostgreSQL using Streamlit secrets
    conn = psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["GMj4jYjVRFebRIwS"],
        port=5432
    )
    return conn

# -----------------------------
# SAVE PREDICTION TO DATABASE
# -----------------------------
def save_prediction(data):
    try:
        conn = get_connection()
        cur = conn.cursor()
        query = """
        INSERT INTO predictions(
            doctor_name, patient_name, patient_id,
            bmi, age, genhlth, physhlth,
            highbp, highchol, physactivity,
            heartdiseaseorattack, diffwalk, smoker,
            risk_score, prediction_date
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cur.execute(query, data)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")

# -----------------------------
# FETCH PREDICTIONS FROM DATABASE
# -----------------------------
def fetch_predictions():
    try:
        conn = get_connection()
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
        st.error(f"Database Fetch Error: {e}")
        return []

# -----------------------------
# LOAD ML MODELS
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("diabetes_full_model.keras")
        encoder = load_model("encoder_model.keras")
        scaler = joblib.load("scaler.pkl")
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, encoder, scaler = load_artifacts()
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
    password = st.text_input("Password", type="password")  # optional
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
    current_date = datetime.now().strftime("%Y-%m-%d")
    st.write(f"Date: **{current_date}**")
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
                val = st.number_input(feature, 10.0, 70.0, 25.0, step=0.1)
            elif feature == "Age":
                val = st.number_input(feature, 1, 120, 30)
            elif feature == "GenHlth":
                val = st.slider(feature, 1, 5, 3)
            elif feature == "PhysHlth":
                val = st.slider(feature, 0, 30, 0)
            else:
                val = st.selectbox(feature, [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            inputs.append(val)
    
    # Predict Button
    if st.button("Predict Diabetes Risk"):
        if patient_name == "" or patient_id == "":
            st.warning("Please enter patient name and ID")
            return
        try:
            data = np.array([inputs], dtype=float)
            scaled = scaler.transform(data)
            encoded = encoder.predict(scaled)
            prediction = model.predict(encoded)
            risk_score = float(prediction[0][0])
            
            st.divider()
            st.subheader("📊 Prediction Result")
            st.write(f"Patient: **{patient_name}**")
            st.write(f"Patient ID: **{patient_id}**")
            st.write(f"Date: **{current_date}**")
            if risk_score > 0.5:
                st.error(f"⚠️ High Diabetes Risk ({risk_score*100:.2f}%)")
            else:
                st.success(f"✅ Low Diabetes Risk ({risk_score*100:.2f}%)")
            st.caption("This is an AI-assisted prediction, not a medical diagnosis.")
            
            # Save prediction to Supabase
            db_data = (
                st.session_state.doctor,
                patient_name,
                patient_id,
                inputs[0], inputs[1], inputs[2], inputs[3],
                inputs[4], inputs[5], inputs[6], inputs[7],
                inputs[8], inputs[9],
                risk_score,
                current_date
            )
            save_prediction(db_data)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    
    # Prediction History Dashboard
    st.divider()
    st.subheader("📂 Prediction History")
    if st.button("Load Prediction History"):
        predictions = fetch_predictions()
        if predictions:
            df = pd.DataFrame(predictions, columns=[
                "Doctor", "Patient Name", "Patient ID", "BMI", "Age", "GenHlth",
                "PhysHlth", "HighBP", "HighChol", "PhysActivity", "HeartDisease",
                "DiffWalk", "Smoker", "Risk Score", "Date"
            ])
            df["Risk Score"] = df["Risk Score"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(df)
        else:
            st.info("No predictions found yet.")

# -----------------------------
# APP CONTROL
# -----------------------------
if st.session_state.logged_in:
    prediction_page()
else:
    login_page()
