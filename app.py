import sqlite3

# -----------------------------
# Create tables
# -----------------------------
def create_tables():
    conn = sqlite3.connect("hospital.db")
    c = conn.cursor()

    # Doctor table
    c.execute("""
    CREATE TABLE IF NOT EXISTS doctors(
        name TEXT,
        national_id TEXT PRIMARY KEY
    )
    """)

    # Patient predictions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        doctor TEXT,
        patient_name TEXT,
        patient_id TEXT,
        date TEXT,
        bmi REAL,
        age INTEGER,
        genhlth INTEGER,
        physhlth INTEGER,
        result REAL
    )
    """)

    conn.commit()
    conn.close()

# -----------------------------
# Add a new doctor
# -----------------------------
def add_doctor(name, national_id):
    conn = sqlite3.connect("hospital.db")
    c = conn.cursor()
    c.execute("INSERT INTO doctors VALUES (?, ?)", (name, national_id))
    conn.commit()
    conn.close()

# -----------------------------
# Doctor login verification
# -----------------------------
def login_doctor(name, national_id):
    conn = sqlite3.connect("hospital.db")
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE name=? AND national_id=?", (name, national_id))
    data = c.fetchone()
    conn.close()
    return data

# -----------------------------
# Save a patient prediction
# -----------------------------
def save_prediction(doctor, patient_name, patient_id, date, bmi, age, genhlth, physhlth, result):
    conn = sqlite3.connect("hospital.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
        (doctor, patient_name, patient_id, date, bmi, age, genhlth, physhlth, result)
    )
    conn.commit()
    conn.close()
