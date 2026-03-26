import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# -----------------------------
# Build the model architecture
# -----------------------------
def build_model():
    model = Sequential([
        Input(shape=(11,)),        # 11 features exactly
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Load or create the model
# -----------------------------
@st.cache_resource
def load_classifier():
    try:
        # Try loading existing Keras 3 model
        model = load_model("diabetes_classifier_new.keras", compile=False)
    except Exception as e:
        st.warning(f"Failed to load model: {e}\nBuilding a new model instead.")
        model = build_model()
    return model

classifier = load_classifier()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Diabetes Risk Prediction")

with st.form("prediction_form"):
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
    Age = st.number_input("Age", 18, 100, 40)
    GenHlth = st.selectbox("General Health", [1, 2, 3, 4, 5], help="1=Excellent 5=Poor")
    PhysHlth = st.number_input("Physical Health (days unhealthy last month)", 0, 30, 0)
    HighBP = st.selectbox("High Blood Pressure", [0, 1])
    HighChol = st.selectbox("High Cholesterol", [0, 1])
    PhysActivity = st.selectbox("Physical Activity (last 30 days)", [0, 1])
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0, 1])
    DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
    Smoker = st.selectbox("Smoker", [0, 1])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input array in the exact same order as training
    input_data = np.array([[BMI, Age, GenHlth, PhysHlth, HighBP, HighChol,
                            PhysActivity, HeartDiseaseorAttack, DiffWalk, Smoker]])
    
    # Make prediction
    prediction = classifier.predict(input_data)[0][0]
    
    st.write(f"Predicted Diabetes Risk: {prediction*100:.2f}%")
