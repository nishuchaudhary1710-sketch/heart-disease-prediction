```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details to get a prediction.")

st.warning(
    "For educational purposes only. "
    "This app is not a medical diagnosis."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"]
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure", 0, 250, 120
    )
    cholesterol = st.number_input(
        "Cholesterol", 0, 700, 200
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar", [0, 1]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"]
    )
    max_hr = st.number_input(
        "Maximum Heart Rate", 50, 250, 150
    )
    exercise_angina = st.selectbox(
        "Exercise Angina", ["N", "Y"]
    )
    oldpeak = st.number_input(
        "Oldpeak", -5.0, 10.0, 0.0, step=0.1
    )
    st_slope = st.selectbox(
        "ST Slope", ["Up", "Flat", "Down"]
    )

if st.button("Predict Heart Disease"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.metric(
        "Model-estimated probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("The model predicts a positive result.")
    else:
        st.success("The model predicts a negative result.")
```
