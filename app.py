# app.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db

# -----------------------------
# Initialize Firebase
# -----------------------------
cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://YOUR_PROJECT_ID.firebaseio.com/'  # Replace with your database URL
})

# -----------------------------
# Firebase Functions
# -----------------------------
def add_doctor(name, national_id):
    ref = db.reference('doctors')
    ref.child(national_id).set({'name': name})

def login_doctor(name, national_id):
    ref = db.reference(f'doctors/{national_id}')
    doctor = ref.get()
    if doctor and doctor.get('name') == name:
        return True
    return False

def save_prediction(doctor, patient_name, patient_id, date, bmi, age, genhlth, physhlth, result):
    ref = db.reference('predictions')
    ref.child(patient_id).set({
        'doctor': doctor,
        'patient_name': patient_name,
        'date': date,
        'bmi': bmi,
        'age': age,
        'genhlth': genhlth,
        'physhlth': physhlth,
        'result': result
    })

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("Diabetes Health Predictor")
    st.divider()

    # --- Doctor login section ---
    st.header("Doctor Login")
    doctor_name = st.text_input("Doctor Name")
    doctor_id = st.text_input("National ID")
    
    if st.button("Login"):
        if login_doctor(doctor_name, doctor_id):
            st.success("Login successful!")
            st.session_state['doctor'] = doctor_name
        else:
            st.error("Invalid credentials. Please add your account first.")

    st.divider()

    # --- Add doctor section (optional) ---
    st.header("Add New Doctor (Admin Only)")
    new_doctor_name = st.text_input("New Doctor Name")
    new_doctor_id = st.text_input("New Doctor National ID")
    if st.button("Add Doctor"):
        if new_doctor_name and new_doctor_id:
            add_doctor(new_doctor_name, new_doctor_id)
            st.success(f"Doctor {new_doctor_name} added successfully!")

    st.divider()

    # --- Patient prediction input ---
    if 'doctor' in st.session_state:
        st.header("Enter Patient Details")
        patient_name = st.text_input("Patient Name")
        patient_id = st.text_input("Patient ID")
        date = st.date_input("Date")
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)
        age = st.number_input("Age", min_value=0, step=1)
        genhlth = st.slider("General Health (1-5)", 1, 5)
        physhlth = st.slider("Physical Health (0-30)", 0, 30)
        result = st.number_input("Predicted Diabetes Risk (0-1)", min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Save Prediction"):
            if patient_name and patient_id:
                save_prediction(st.session_state['doctor'], patient_name, patient_id, str(date),
                                bmi, age, genhlth, physhlth, result)
                st.success(f"Prediction for {patient_name} saved successfully!")
            else:
                st.error("Please enter patient name and ID.")

if __name__ == "__main__":
    main()
