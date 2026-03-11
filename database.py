import firebase_admin
from firebase_admin import credentials, db

# -----------------------------
# Initialize Firebase
# -----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://YOUR_PROJECT_ID.firebaseio.com/'  # replace with your Firebase Realtime Database URL
})

# -----------------------------
# Add a new doctor
# -----------------------------
def add_doctor(name, national_id):
    ref = db.reference('doctors')
    ref.child(national_id).set({
        'name': name
    })

# -----------------------------
# Doctor login verification
# -----------------------------
def login_doctor(name, national_id):
    ref = db.reference(f'doctors/{national_id}')
    doctor = ref.get()
    if doctor and doctor.get('name') == name:
        return True
    return False

# -----------------------------
# Save a patient prediction
# -----------------------------
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
# Example usage
# -----------------------------
# add_doctor("Dr. Smith", "12345")
# print(login_doctor("Dr. Smith", "12345"))
# save_prediction("Dr. Smith", "John Doe", "p001", "2026-03-11", 28.3, 45, 3, 2, 0.85)
