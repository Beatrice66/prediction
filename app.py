import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

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
# MODEL PATHS
# =====================================
SCALER_PATH = "scaler.pkl"
CLASSIFIER_PATH = "diabetes_classifier.keras"

# =====================================
# LOAD OR CREATE MODEL
# =====================================
@st.cache_resource
def load_or_create_model():
    # Load scaler if exists
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = StandardScaler()  # will fit later

    # Load classifier if exists
    if os.path.exists(CLASSIFIER_PATH):
        try:
            classifier = load_model(CLASSIFIER_PATH, compile=False)
            st.info("Loaded existing model.")
        except Exception as e:
            st.warning(f"Failed to load existing model, creating a new one. {e}")
            classifier = None
    else:
        classifier = None

    # If classifier doesn't exist, create a new Keras 3 model
    if classifier is None:
        st.info("Creating a new classifier model...")
        classifier = Sequential([
            Dense(16, activation="relu", input_shape=(10,)),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        classifier.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        st.success("New model created. Remember to train it before using predictions!")

    return scaler, classifier

scaler, classifier = load_or_create_model()

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
    GenHlth = st.selectbox("General Health (1=Excellent, 5=Poor)", [1,2,3,4,5])
    PhysHlth = st.number_input("Physical Health (days unhealthy last month)", 0, 30, 0)
    HighBP = st.selectbox("High Blood Pressure", [0,1])
    HighChol = st.selectbox("High Cholesterol", [0,1])
    PhysActivity = st.selectbox("Physical Activity (last 30 days)", [0,1])
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0,1])
    DiffWalk = st.selectbox("Difficulty Walking", [0,1])
    Smoker = st.selectbox("Smoker", [0,1])
    
    submitted = st.form_submit_button("Predict")

# =====================================
# PREDICTION
# =====================================
if submitted:
    try:
        input_data = np.array([[BMI, Age, GenHlth, PhysHlth, HighBP, HighChol,
                                PhysActivity, HeartDiseaseorAttack, DiffWalk, Smoker]])

        # If scaler is fitted, scale the input
        if hasattr(scaler, "mean_"):
            scaled_data = scaler.transform(input_data)
        else:
            scaled_data = input_data
            scaler.fit(input_data)
            joblib.dump(scaler, SCALER_PATH)

        # Make prediction
        prediction = classifier.predict(scaled_data)
        probability = float(prediction[0][0])
        risk = "High Risk" if probability > 0.5 else "Low Risk"

        st.subheader("Prediction Result")
        st.write(f"Risk Level: **{risk}**")
        st.write(f"Probability: **{probability*100:.2f}%**")

        # Save prediction to database
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions(
                bmi, age, genhlth, physhlth, highbp, highchol,
                physactivity, heartdiseaseorattack, diffwalk, smoker,
                prediction, probability, created_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                BMI, Age, GenHlth, PhysHlth, HighBP, HighChol,
                PhysActivity, HeartDiseaseorAttack, DiffWalk, Smoker,
                risk, probability, datetime.now()
            )
        )
        conn.commit()
        cur.close()
        conn.close()
        st.success("Prediction saved successfully!")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =====================================
# DASHBOARD
# =====================================
st.subheader("Prediction History")
try:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df)
        st.bar_chart(df["prediction"].value_counts())
    else:
        st.write("No predictions yet.")
except Exception as e:
    st.warning(f"Database not connected: {e}")
