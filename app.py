import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# =====================================
# DATABASE CONNECTION
# =====================================

def get_connection():
    conn = psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        port="5432"
    )
    return conn


# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    try:
        scaler = joblib.load("scaler.pkl")
    except:
        st.error("Scaler file not found or incompatible")
        scaler = None

    try:
        encoder = load_model("encoder_model.keras", compile=False)
    except:
        st.error("Encoder model could not be loaded")
        encoder = None

    try:
        classifier = load_model("diabetes_classifier.keras", compile=False)
    except:
        st.error("Classifier model could not be loaded")
        classifier = None

    return scaler, encoder, classifier


scaler, encoder, classifier = load_models()

# =====================================
# PAGE TITLE
# =====================================

st.title("Diabetes Risk Prediction")
st.write("Enter patient information below to predict diabetes risk.")

# =====================================
# INPUT FORM
# =====================================

with st.form("prediction_form"):

    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)

    Age = st.number_input("Age", 18, 100, 40)

    GenHlth = st.selectbox(
        "General Health",
        [1,2,3,4,5],
        help="1=Excellent 5=Poor"
    )

    PhysHlth = st.number_input(
        "Physical Health (days unhealthy last month)",
        0,
        30,
        0
    )

    HighBP = st.selectbox("High Blood Pressure", [0,1])

    HighChol = st.selectbox("High Cholesterol", [0,1])

    PhysActivity = st.selectbox(
        "Physical Activity (last 30 days)",
        [0,1]
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Attack",
        [0,1]
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking",
        [0,1]
    )

    Smoker = st.selectbox(
        "Smoker",
        [0,1]
    )

    submitted = st.form_submit_button("Predict")


# =====================================
# PREDICTION
# =====================================

if submitted:

    if scaler is None or encoder is None or classifier is None:
        st.error("Model files are missing. Please check deployment.")
    else:

        try:

            input_data = np.array([[

                BMI,
                Age,
                GenHlth,
                PhysHlth,
                HighBP,
                HighChol,
                PhysActivity,
                HeartDiseaseorAttack,
                DiffWalk,
                Smoker

            ]])

            # SCALE
            scaled = scaler.transform(input_data)

            # ENCODE
            encoded = encoder.predict(scaled)

            # CLASSIFY
            prediction = classifier.predict(encoded)

            probability = float(prediction[0][0])

            risk = "High Risk" if probability > 0.5 else "Low Risk"

            st.subheader("Prediction Result")

            st.write(f"Risk Level: **{risk}**")

            st.write(f"Probability: **{probability*100:.2f}%**")


            # =====================================
            # SAVE TO DATABASE
            # =====================================

            conn = get_connection()

            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO predictions(
                bmi,
                age,
                genhlth,
                physhlth,
                highbp,
                highchol,
                physactivity,
                heartdiseaseorattack,
                diffwalk,
                smoker,
                prediction,
                probability,
                created_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    BMI,
                    Age,
                    GenHlth,
                    PhysHlth,
                    HighBP,
                    HighChol,
                    PhysActivity,
                    HeartDiseaseorAttack,
                    DiffWalk,
                    Smoker,
                    risk,
                    probability,
                    datetime.now()
                )
            )

            conn.commit()

            cur.close()

            conn.close()

            st.success("Prediction saved successfully")

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# =====================================
# DASHBOARD
# =====================================

st.subheader("Prediction History")

try:

    conn = get_connection()

    df = pd.read_sql(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50",
        conn
    )

    conn.close()

    if not df.empty:

        st.dataframe(df)

        st.bar_chart(df["prediction"].value_counts())

    else:

        st.write("No predictions yet.")

except:

    st.warning("Database not connected.")
