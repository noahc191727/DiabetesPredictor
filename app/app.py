import sys
import os

# Add project root so src/ imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.inference import predict


# -----------------------
# Streamlit Page Settings
# -----------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)


# -----------------------
# Header
# -----------------------
st.title("🩺 Diabetes Risk Prediction Tool")

st.markdown("""
Welcome!  
This app predicts the probability of **Type 2 Diabetes** using a machine-learning model trained on health indicators.

Just enter the patient’s values below ⬇️
""")

st.divider()


# -----------------------
# Input Section
# -----------------------
st.subheader("🔢 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=100)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=80.0, value=28.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)

st.divider()


# -----------------------
# Put values into dict for preprocessing
# -----------------------
user_values = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}


# -----------------------
# Prediction Button
# -----------------------
st.markdown("### 📈 Run Prediction")

if st.button("Predict Diabetes Risk", type="primary"):
    pred, prob = predict(user_values)

    st.divider()
    st.subheader("📊 Prediction Results")

    # Probability formatting
    percent = round(prob * 100, 1)

    # Risk labeling
    if prob < 0.25:
        category = "🟢 Low Risk"
        color = "green"
    elif prob < 0.55:
        category = "🟡 Moderate Risk"
        color = "orange"
    else:
        category = "🔴 High Risk"
        color = "red"

    # Styled output
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5;">
        <h3 style="color:{color};">{category}</h3>
        <p style="font-size:22px; margin:0;"><strong>Estimated Probability:</strong> {percent}%</p>
    </div>
    """, unsafe_allow_html=True)

    # -------- Risk Meter (progress bar) --------
    st.write("### 📍 Risk Meter")
    st.progress(prob)

    st.divider()

    # -------- Interpretation Section --------
    st.subheader("📘 Model Interpretation & Notes")

    st.write("""
    **How to interpret this prediction:**  
    - The probability is computed using a calibrated **XGBoost classifier**.  
    - Inputs are preprocessed using the *exact same feature engineering* as in training:
        - BMI categories  
        - Insulin normal/abnormal  
        - Glucose category (low/normal/prediabetic/high)  
    - Numerical features are scaled via **StandardScaler**.  
    - The final output represents the probability that the patient has diabetes **based on their input values only**.
    """)

    st.divider()

    # -------- Model Info --------
    st.subheader("🧠 Model Information")

    st.markdown("""
    - **Model:** Calibrated XGBoost  
    - **Training:** Performed in Jupyter Notebook with full EDA  
    - **Calibration:** Isotonic regression  
    - **Features:**  
        - Pregnancies  
        - Glucose  
        - Blood Pressure  
        - Skin Thickness  
        - Insulin  
        - BMI + BMI Category  
        - Diabetes Pedigree Function  
        - Age  
        - Glucose Category  
        - Insulin Category  
    """)

else:
    st.info("Click **Predict Diabetes Risk** to generate a prediction.")
