import streamlit as st
import numpy as np
import keras
import os
from database import *

st.set_page_config(page_title="Hospital Diabetes AI", layout="centered")

create_tables()

# Session
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if "doctor" not in st.session_state:
    st.session_state.doctor=""

# Load model
@st.cache_resource
def load_model():
    path=os.path.join(os.path.dirname(__file__),"diabetes_full_model.keras")
    return keras.models.load_model(path)

model=load_model()

# LOGIN PAGE
def login():

    st.title("🔐 Doctor Login")

    name=st.text_input("Full Name (Two Names)")
    national_id=st.text_input("National ID",type="password")

    if st.button("Login"):

        data=login_doctor(name,national_id)

        if data:
            st.session_state.logged_in=True
            st.session_state.doctor=name
            st.success("Login successful")
            st.rerun()

        else:
            st.error("Doctor not registered")

# REGISTER DOCTOR
def register():

    st.subheader("Register New Doctor")

    name=st.text_input("Full Name")
    national_id=st.text_input("National ID")

    if st.button("Register Doctor"):

        add_doctor(name,national_id)

        st.success("Doctor registered")

# MAIN APP
def predictor():

    st.title("🩺 Diabetes Risk Predictor")

    st.write(f"Logged in as **Dr. {st.session_state.doctor}**")

    if st.button("Logout"):
        st.session_state.logged_in=False
        st.rerun()

    with st.form("patient_form"):

        bmi=st.number_input("BMI",10.0,70.0,25.0)

        age=st.number_input("Age",1,120,30)

        genhlth=st.slider("General Health",1,5,3)

        physhlth=st.slider("Physical Health Bad Days",0,30,0)

        highbp=st.selectbox("High Blood Pressure",[0,1])

        highchol=st.selectbox("High Cholesterol",[0,1])

        activity=st.selectbox("Physical Activity",[0,1])

        heart=st.selectbox("Heart Disease",[0,1])

        diffwalk=st.selectbox("Difficulty Walking",[0,1])

        smoker=st.selectbox("Smoker",[0,1])

        submit=st.form_submit_button("Predict")

    if submit:

        data=np.array([[bmi,age,genhlth,physhlth,highbp,
                        highchol,activity,heart,diffwalk,smoker]],
                        dtype="float32")

        pred=model.predict(data,verbose=0)

        risk=float(pred[0][0])

        st.divider()

        if risk>0.5:
            st.error(f"High Risk ({risk*100:.1f}%)")
        else:
            st.success(f"Low Risk ({risk*100:.1f}%)")

        save_prediction(st.session_state.doctor,bmi,age,genhlth,physhlth,risk)

# APP CONTROL
if st.session_state.logged_in:
    predictor()
else:
    login()
    register()
