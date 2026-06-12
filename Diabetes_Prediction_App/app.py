import streamlit as st
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "diabetes_model.pkl"), "rb")
)

scaler = pickle.load(
    open(os.path.join(BASE_DIR, "scaler.pkl"), "rb")
)

st.title("Diabetes Prediction Web App")

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0
)
Age = st.number_input("Age", min_value=0)

if st.button("Predict"):

    input_data = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 0:
        st.success("The person is NOT diabetic")
    else:
        st.error("The person IS diabetic")