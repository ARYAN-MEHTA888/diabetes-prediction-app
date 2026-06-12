import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "diabetes_model.pkl"), "rb")
)

scaler = pickle.load(
    open(os.path.join(BASE_DIR, "scaler.pkl"), "rb")
)

# Sidebar
st.sidebar.title("🩺 About")
st.sidebar.info("""
This Diabetes Prediction System uses
Machine Learning (SVM) to predict whether
a person is likely to have diabetes.

Dataset:
Pima Indians Diabetes Dataset
""")

# Title
st.title("🩺 Diabetes Prediction System")
st.markdown(
    "Enter patient health information below."
)

# Metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "SVM")

with col2:
    st.metric("Accuracy", "78%")

st.markdown("---")

# Inputs
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

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Low Risk of Diabetes")
    else:
        st.error("⚠️ High Risk of Diabetes")

        st.warning("""
        Please consult a healthcare professional.
        This prediction is for educational purposes only.
        """)

    # Chart
    chart_data = pd.DataFrame({
        "Feature": ["Glucose", "BMI", "Age"],
        "Value": [Glucose, BMI, Age]
    })

    st.subheader("Patient Summary")
    st.bar_chart(
        chart_data.set_index("Feature")
    )

st.markdown("---")
st.markdown(
    "Developed by Aryan Mehta | Machine Learning Project"
)
