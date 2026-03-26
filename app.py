import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime


# -------------------------
# DATABASE CONNECTION
# -------------------------
def get_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        port=5432
    )


# -------------------------
# SAVE PREDICTION
# -------------------------
def save_prediction(data):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO predictions(
        doctor_name, patient_name, patient_id,
        bmi, age, genhlth, physhlth,
        highbp, highchol, physactivity,
        heartdiseaseorattack, diffwalk, smoker,
        risk_score, prediction_date)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data)

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        st.error(f"Database Error: {e}")


# -------------------------
# LOAD HISTORY
# -------------------------
def load_history():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM predictions ORDER BY prediction_date DESC")

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return rows

    except Exception as e:
        st.error(e)
        return []


# -------------------------
# LOAD MODEL FILES
# -------------------------
@st.cache_resource
def load_models():

    model = load_model("diabetes_full_model.keras")
    encoder = load_model("encoder_model.keras")
    scaler = joblib.load("scaler.pkl")

    return model, encoder, scaler


model, encoder, scaler = load_models()


# -------------------------
# LOGIN SESSION
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# -------------------------
# LOGIN PAGE
# -------------------------
def login():

    st.title("Doctor Login")

    doctor = st.text_input("Doctor Name")

    if st.button("Login"):

        if doctor == "":
            st.warning("Enter doctor name")

        else:
            st.session_state.logged_in = True
            st.session_state.doctor = doctor
            st.rerun()


# -------------------------
# MAIN APP
# -------------------------
def app():

    st.title("Diabetes Risk Prediction")

    st.write("Logged in as:", st.session_state.doctor)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    st.subheader("Patient Details")

    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

    st.divider()

    st.subheader("Health Inputs")

    bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    age = st.number_input("Age", 1, 120, 30)

    genhlth = st.slider("General Health", 1, 5, 3)
    physhlth = st.slider("Physical Health Days", 0, 30, 0)

    highbp = st.selectbox("High Blood Pressure", [0,1])
    highchol = st.selectbox("High Cholesterol", [0,1])
    physactivity = st.selectbox("Physical Activity", [0,1])
    heartdisease = st.selectbox("Heart Disease", [0,1])
    diffwalk = st.selectbox("Difficulty Walking", [0,1])
    smoker = st.selectbox("Smoker", [0,1])


    if st.button("Predict Diabetes Risk"):

        try:

            # -------------------------
            # CREATE INPUT ARRAY
            # -------------------------
            features = np.array([
                bmi, age, genhlth, physhlth,
                highbp, highchol, physactivity,
                heartdisease, diffwalk, smoker
            ], dtype=float)

            features = features.reshape(1,10)

            # -------------------------
            # SCALE
            # -------------------------
            scaled = scaler.transform(features)

            # -------------------------
            # ENCODER
            # -------------------------
            encoded = encoder.predict(scaled)

            encoded = np.array(encoded)

            # -------------------------
            # SHAPE FIX
            # -------------------------
            if encoded.shape[1] != 10:

                # expand to required size
                fixed = np.zeros((1,10))
                length = min(encoded.shape[1],10)

                fixed[0,:length] = encoded[0,:length]

                encoded = fixed

            # -------------------------
            # MODEL PREDICTION
            # -------------------------
            prediction = model.predict(encoded)

            risk_score = float(prediction[0][0])

            st.subheader("Prediction Result")

            st.write("Risk Score:", risk_score)

            if risk_score > 0.5:
                st.error(f"High Risk ({risk_score*100:.2f}%)")
            else:
                st.success(f"Low Risk ({risk_score*100:.2f}%)")


            # -------------------------
            # SAVE TO DATABASE
            # -------------------------
            data = (
                st.session_state.doctor,
                patient_name,
                patient_id,
                bmi, age, genhlth, physhlth,
                highbp, highchol, physactivity,
                heartdisease, diffwalk, smoker,
                risk_score,
                datetime.now().strftime("%Y-%m-%d")
            )

            save_prediction(data)

        except Exception as e:
            st.error(f"Prediction Error: {e}")


    st.divider()

    st.subheader("Prediction History")

    if st.button("Load History"):

        rows = load_history()

        if rows:

            df = pd.DataFrame(rows)

            st.dataframe(df)

        else:
            st.info("No predictions yet")


# -------------------------
# ROUTER
# -------------------------
if st.session_state.logged_in:
    app()
else:
    login()
