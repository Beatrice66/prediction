import sqlite3

DB_NAME = "diabetes_app.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bmi REAL,
        age INTEGER,
        genhlth INTEGER,
        physhlth INTEGER,
        highbp INTEGER,
        highchol INTEGER,
        physactivity INTEGER,
        heartdisease INTEGER,
        diffwalk INTEGER,
        smoker INTEGER,
        probability REAL,
        prediction INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patients (
            bmi, age, genhlth, physhlth,
            highbp, highchol, physactivity,
            heartdisease, diffwalk, smoker,
            probability, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()