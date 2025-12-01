import os
import sys
import streamlit as st
from src.inference import predict

# Add project root for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Diabetes Risk Prediction Tool")

st.markdown(
    """
Welcome!
This app predicts the probability of **Type 2 Diabetes** using a
machine-learning model trained on health indicators.
Enter patient values below ⬇️
"""
)

st.divider()

st.subheader("🔢 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)", min_value=0, max_value=300, value=120
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=0, max_value=200, value=70
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=0, max_value=100, value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (µU/mL)", min_value=0, max_value=900, value=100
    )
    bmi = st.number_input(
        "Body Mass Index (BMI)", min_value=0.0, max_value=80.0, value=28.0
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=35
    )

st.divider()

user_values = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}

st.markdown("### 📈 Run Prediction")

if st.button("Predict Diabetes Risk", type="primary"):
    pred, prob = predict(user_values)

    st.divider()
    st.subheader("📊 Prediction Results")

    percent = round(prob * 100, 1)

    if prob < 0.25:
        category = "🟢 Low Risk"
        color = "green"
    elif prob < 0.55:
        category = "🟡 Moderate Risk"
        color = "orange"
    else:
        category = "🔴 High Risk"
        color = "red"

    st.markdown(
        f"""
    <div style="padding: 20px; border-radius: 
    10px; background-color: #f5f5f5;">
        <h3 style="color:{color};">{category}</h3>
        <p style="font-size:22px; margin:0;">
            <strong>Estimated Probability:</strong> {percent}%
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("### 📍 Risk Meter")
    st.progress(prob)

    st.divider()

    st.subheader("📘 Model Interpretation & Notes")
    st.write(
        """
**How to interpret this prediction:**
- The probability comes from a calibrated **XGBoost classifier**.
- Inputs are processed using the same feature engineering as training.
- Features include BMI category, glucose ranges, insulin category, and more.
"""
    )

    st.divider()

    st.subheader("🧠 Model Information")
    st.markdown(
        """
- **Model:** Calibrated XGBoost
- **Training:** Full EDA + preprocessing
- **Calibration:** Isotonic regression
"""
    )

else:
    st.info("Click **Predict Diabetes Risk** to generate a prediction.")
