import streamlit as st
import numpy as np
import pandas as pd
<<<<<<< HEAD
import psycopg2
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime


# =====================================
# DATABASE CONNECTION (SUPABASE)
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

    scaler = joblib.load("scaler.pkl")

    encoder = load_model(
        "encoder_model.keras",
        compile=False
    )

    classifier = load_model(
        "diabetes_classifier.keras",
        compile=False
    )

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

except Exception as e:

    st.warning("Database not connected.")
=======
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

st.title("Diabetes Risk Prediction")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    # load keras model safely
    model = load_model("diabetes_full_model.keras", compile=False)

    # load encoder + scaler
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, encoder, scaler


model, encoder, scaler = load_models()


# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
glucose = st.number_input("Glucose Level", 50, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 40, 200, 80)
insulin = st.number_input("Insulin", 0, 900, 80)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

pregnancies = st.number_input("Pregnancies", 0, 20, 1)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    try:

        # categorical dataframe
        cat_df = pd.DataFrame({
            "Gender":[gender],
            "Smoking":[smoking],
            "FamilyHistory":[family_history]
        })

        # encode categorical
        encoded = encoder.transform(cat_df)

        # numeric features
        num_features = np.array([[

            age,
            bmi,
            glucose,
            blood_pressure,
            insulin,
            skin_thickness,
            pregnancies

        ]])

        # scale numeric
        num_scaled = scaler.transform(num_features)

        # combine
        final_input = np.concatenate([num_scaled, encoded], axis=1)

        # ensure correct shape
        final_input = final_input.reshape(1,10)

        # prediction
        prediction = model.predict(final_input)

        prob = float(prediction[0][0])

        st.write("Prediction Probability:", prob)

        if prob > 0.5:
            st.error("High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
>>>>>>> 437607cdcbc78f841be9abd85d64d71487da40bb
