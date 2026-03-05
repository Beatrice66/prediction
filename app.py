import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import sqlite3
st.write("App started successfully

# ---------------------------
# DATABASE
# ---------------------------
conn = sqlite3.connect("mydatabase.db", check_same_thread=False)
c = conn.cursor()

def create_table():
    c.execute("""
    CREATE TABLE IF NOT EXISTS diabetes_data(
        BMI REAL,
        Age INTEGER,
        GenHlth INTEGER,
        PhysHlth INTEGER,
        HighBP INTEGER,
        Result TEXT
    )
    """)

def add_data(BMI, Age, GenHlth, PhysHlth, HighBP, Result):
    c.execute("INSERT INTO diabetes_data VALUES (?,?,?,?,?,?)",
              (BMI, Age, GenHlth, PhysHlth, HighBP, Result))
    conn.commit()

create_table()

# ---------------------------
# LOAD MODEL
# ---------------------------
model = tf.keras.models.load_model("diabetes_full_model.keras")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Diabetes Risk Prediction App")

st.write("Enter health information below to predict diabetes risk.")

# Inputs
BMI = st.number_input("BMI", 10.0, 60.0)
Age = st.number_input("Age", 18, 100)
GenHlth = st.selectbox("General Health (1=Excellent, 5=Poor)", [1,2,3,4,5])
PhysHlth = st.number_input("Physical Health (Days unhealthy)", 0, 30)
HighBP = st.selectbox("High Blood Pressure", [0,1])

# Prediction button
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[BMI, Age, GenHlth, PhysHlth, HighBP]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    risk = prediction[0][0]

    if risk > 0.5:
        result = "High Risk of Diabetes"
        st.error(result)
    else:
        result = "Low Risk of Diabetes"
        st.success(result)

    add_data(BMI, Age, GenHlth, PhysHlth, HighBP, result)