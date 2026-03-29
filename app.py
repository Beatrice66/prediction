import psycopg2
import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
from fpdf import FPDF

# --- 1. SECURE CONFIG ---
DB_CONFIG = {
    "host": "localhost",
    "database": "diabetes_app",
    "user": "postgres",
    "password": "38744474", 
    "port": 5432
}

def get_connection():
    # Pass parameters explicitly to prevent 'no password supplied' errors
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        port=DB_CONFIG["port"]
    )

def save_prediction(data):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Ensure 17 %s placeholders match your table: doctor, p_name, p_id, p_phone, p_loc, 
        # bmi, age, gh, ph, bp, chol, act, heart, walk, smoke, prob, date
        query = """
        INSERT INTO predictions(
            doctor_name, patient_name, patient_id, patient_phone, patient_location,
            bmi, age, genhlth, physhlth, highbp, highchol, 
            physactivity, heartdiseaseorattack, diffwalk, smoker,
            risk_score, prediction_date
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cur.execute(query, data)
        conn.commit()
        st.toast("✔️ Patient Data Synced to Hospital Registry")
    except Exception as e:
        st.error(f"❌ Database Flow Error: {e}")
    finally:
        if conn: conn.close()

# --- 2. AI ASSETS ---
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("modeled(1).keras")
        encoder = load_model("encodered_model.keras")
        scaler = joblib.load("scaler(1).pkl")
        return model, encoder, scaler
    except Exception as e:
        st.error(f"❌ AI Assets Offline: {e}")
        return None, None, None

model, encoder, scaler = load_artifacts()

# --- 3. PDF GENERATOR ---
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, "MEDVANTAGE HOSPITAL ASSESSMENT", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Attending Physician: Dr. {data['doctor']}", ln=True)
    pdf.cell(0, 10, f"Patient Name: {data['name']} (ID: {data['id']})", ln=True)
    pdf.cell(0, 10, f"Contact: {data['phone']} | Location: {data['loc']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    risk = "HIGH" if data['prob'] > 0.5 else "LOW"
    pdf.cell(0, 10, f"DIABETES RISK: {risk} ({round(data['prob']*100, 2)}%)", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. DASHBOARD UI ---
def main_portal():
    st.sidebar.title(f"👨‍⚕️ Dr. {st.session_state.doctor}")
    page = st.sidebar.selectbox("Hospital Menu", ["Intake & Analysis", "Patient Registry"])

    if page == "Intake & Analysis":
        st.header("📋 New Patient Clinical Intake")
        with st.container():
            c1, c2 = st.columns(2)
            p_name = c1.text_input("Full Name")
            p_id = c2.text_input("National ID")
            p_phone = c1.text_input("Phone Number")
            p_loc = c2.text_input("Residential Area")

        st.divider()
        st.subheader("Clinical Vital Signs")
        v1, v2 = st.columns(2)
        with v1:
            bmi = v1.number_input("BMI", 10.0, 70.0, 25.0)
            age = v1.slider("Age Category", 1, 13, 5)
            gh = v1.slider("Gen Health (1-5)", 1, 5, 2)
            ph = v1.number_input("Days Physically Ill", 0, 30, 0)
        with v2:
            bp = v2.selectbox("High BP", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            chol = v2.selectbox("High Chol", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            act = v2.selectbox("Phys Activity", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            heart = v2.selectbox("Heart Disease", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            walk = v2.selectbox("Diff Walk", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            smoke = v2.selectbox("Smoker", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

        if st.button("🚀 Analyze, Save & Generate Report", use_container_width=True):
            if not p_name or not p_id:
                st.warning("Identification fields are mandatory.")
            else:
                # 1. AI Logic
                input_data = [bmi, age, gh, ph, bp, chol, act, heart, walk, smoke]
                scaled = scaler.transform([input_data])
                encoded = encoder.predict(scaled)
                prob = float(model.predict(encoded)[0][0])
                
                # 2. UI Result
                if prob > 0.5: st.error(f"HIGH RISK: {prob*100:.2f}%")
                else: st.success(f"LOW RISK: {prob*100:.2f}%")

                # 3. DB Save
                entry = (st.session_state.doctor, p_name, p_id, p_phone, p_loc,
                         bmi, age, gh, ph, bp, chol, act, heart, walk, smoke, 
                         prob, datetime.now().date())
                save_prediction(entry)

                # 4. Report
                pdf_data = {"doctor":st.session_state.doctor, "name":p_name, "id":p_id, "phone":p_phone, "loc":p_loc, "prob":prob}
                pdf_bytes = generate_pdf(pdf_data)
                st.download_button("📥 Download Official Patient Report", pdf_bytes, f"Report_{p_id}.pdf")

    elif page == "Patient Registry":
        st.header("🗂️ Hospital Patient Registry")
        q = st.text_input("🔎 Search by Name or ID")
        try:
            with get_connection() as conn:
                df = pd.read_sql("SELECT * FROM predictions ORDER BY prediction_date DESC", conn)
                if q:
                    df = df[df['patient_name'].str.contains(q, case=False) | df['patient_id'].str.contains(q)]
                st.dataframe(df[['patient_name', 'patient_id', 'patient_phone', 'patient_location', 'risk_score', 'prediction_date']], use_container_width=True)
        except Exception as e:
            st.error(f"Registry Access Failed: {e}")

# --- 5. AUTH & RUN ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False

if st.session_state.logged_in:
    main_portal()
else:
    st.title("🏥 MedVantage Login")
    name = st.text_input("Physician Full Name")
    if st.button("Access System"):
        if name:
            st.session_state.logged_in = True
            st.session_state.doctor = name
            st.rerun()
