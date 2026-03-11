import streamlit as st
import numpy as np
import keras
import osimport streamlit as st
import numpy as np
import keras
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

# -----------------------------
# Example Doctor Accounts
# (Name : National ID)
# -----------------------------
users = {
    "John Kamau": "12345678",
    "Mary Wanjiku": "23456789",
    "Peter Otieno": "34567890"
}

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# Login Page
# -----------------------------
def login():

    st.title("🔐 Doctor Login")
    st.write("Enter your **Full Name** and **National ID** to access the system.")

    fullname = st.text_input("Full Name (First and Last Name)")
    national_id = st.text_input("National ID", type="password")

    if st.button("Login"):

        if fullname in users and users[fullname] == national_id:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()

        else:
            st.error("Invalid name or National ID")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_diabetes_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_full_model.keras")
    return keras.models.load_model(model_path)

# -----------------------------
# Diabetes Predictor Page
# -----------------------------
def diabetes_app():

    model = load_diabetes_model()

    st.title("🩺 Diabetes Health Predictor")
    st.write("Doctor: Please enter patient health information.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    feature_names = [
        "BMI",
        "Age",
        "GenHlth",
        "PhysHlth",
        "HighBP",
        "HighChol",
        "PhysActivity",
        "HeartDiseaseorAttack",
        "DiffWalk",
        "Smoker"
    ]

    with st.form("health_form"):

        cols = st.columns(2)
        user_inputs = []

        for i, name in enumerate(feature_names):

            with cols[i % 2]:

                if name == "BMI":
                    val = st.number_input(
                        "BMI",
                        min_value=10.0,
                        max_value=70.0,
                        value=25.0,
                        step=0.1
                    )

                elif name == "Age":
                    val = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=30
                    )

                elif name == "GenHlth":
                    val = st.slider(
                        "General Health",
                        1,
                        5,
                        3,
                        help="1 = Excellent, 5 = Poor"
                    )

                elif name == "PhysHlth":
                    val = st.slider(
                        "Physical Health (Bad Days)",
                        0,
                        30,
                        0,
                        help="Days physical health was not good in past 30 days"
                    )

                else:
                    val = st.selectbox(
                        name,
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )

                user_inputs.append(val)

        submit = st.form_submit_button("Predict Diabetes Risk")

    if submit:

        data = np.array([user_inputs], dtype="float32")

        prediction = model.predict(data, verbose=0)

        risk_score = float(prediction[0][0])

        st.divider()

        if risk_score > 0.5:
            st.error("⚠️ High Diabetes Risk")
            st.write(f"Probability: **{risk_score*100:.1f}%**")
        else:
            st.success("✅ Low Diabetes Risk")
            st.write(f"Probability: **{risk_score*100:.1f}%**")

        st.caption("This system assists doctors and is not a medical diagnosis.")

# -----------------------------
# App Control
# -----------------------------
if st.session_state.logged_in:
    diabetes_app()
else:
    login()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Diabetes Health Predictor", layout="centered")

# -----------------------------
# Example Doctor Accounts
# (Replace with database later)
# -----------------------------
users = {
    "doctor1": "1001",
    "doctor2": "1002",
    "doctor3": "1003"
}

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# Login Function
# -----------------------------
def login():

    st.title("🔐 Doctor Login")

    username = st.text_input("Username")
    userid = st.text_input("User ID", type="password")

    if st.button("Login"):

        if username in users and users[username] == userid:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()

        else:
            st.error("Invalid username or user ID")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_diabetes_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_full_model.keras")
    return keras.models.load_model(model_path)

# -----------------------------
# Diabetes Prediction Page
# -----------------------------
def diabetes_app():

    model = load_diabetes_model()

    st.title("🩺 Diabetes Health Predictor")
    st.write("Doctor: Enter patient health information below")

    # Logout Button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    feature_names = [
        "BMI",
        "Age",
        "GenHlth",
        "PhysHlth",
        "HighBP",
        "HighChol",
        "PhysActivity",
        "HeartDiseaseorAttack",
        "DiffWalk",
        "Smoker"
    ]

    with st.form("health_form"):

        cols = st.columns(2)
        user_inputs = []

        for i, name in enumerate(feature_names):

            with cols[i % 2]:

                if name == "BMI":
                    val = st.number_input(
                        "BMI",
                        min_value=10.0,
                        max_value=70.0,
                        value=25.0,
                        step=0.1
                    )

                elif name == "Age":
                    val = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=30
                    )

                elif name == "GenHlth":
                    val = st.slider(
                        "General Health",
                        1,
                        5,
                        3,
                        help="1 = Excellent, 5 = Poor"
                    )

                elif name == "PhysHlth":
                    val = st.slider(
                        "Physical Health (Bad Days)",
                        0,
                        30,
                        0,
                        help="Number of days health was not good in past 30 days"
                    )

                else:
                    val = st.selectbox(
                        name,
                        options=[0,1],
                        format_func=lambda x: "Yes" if x==1 else "No"
                    )

                user_inputs.append(val)

        submit = st.form_submit_button("Predict Diabetes Risk")

    if submit:

        data = np.array([user_inputs], dtype="float32")

        prediction = model.predict(data, verbose=0)

        risk_score = float(prediction[0][0])

        st.divider()

        if risk_score > 0.5:
            st.error("⚠️ High Diabetes Risk")
            st.write(f"Probability: **{risk_score*100:.1f}%**")
        else:
            st.success("✅ Low Diabetes Risk")
            st.write(f"Probability: **{risk_score*100:.1f}%**")

        st.caption("This tool assists doctors and is not a medical diagnosis.")

# -----------------------------
# App Control
# -----------------------------
if st.session_state.logged_in:
    diabetes_app()
else:
    login()
