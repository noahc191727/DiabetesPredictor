import numpy as np
import pandas as pd


def preprocess_input(data: dict, scaler):
    """
    Preprocesses a single patient input dictionary using the exact
    feature engineering and one-hot encoding used during training.
    """

    # Convert to DataFrame
    df = pd.DataFrame([data])


    # ---------------------------
    # Feature Engineering (match training exactly)
    # ---------------------------

    # BMI category
    bmi = df.loc[0, "BMI"]
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi <= 24.9:
        bmi_cat = "Normal"
    elif bmi <= 29.9:
        bmi_cat = "Overweight"
    elif bmi <= 34.9:
        bmi_cat = "Obesity 1"
    elif bmi <= 39.9:
        bmi_cat = "Obesity 2"
    else:
        bmi_cat = "Obesity 3"

    # Insulin category
    ins = df.loc[0, "Insulin"]
    ins_cat = "Normal" if 16 <= ins <= 166 else "Abnormal"

    # Glucose category
    g = df.loc[0, "Glucose"]
    if g <= 70:
        glc_cat = "Low"
    elif g <= 99:
        glc_cat = "Normal"
    elif g <= 125:
        glc_cat = "Prediabetic"
    else:
        glc_cat = "High"

    df["NewBMI"] = bmi_cat
    df["NewInsulinScore"] = ins_cat
    df["NewGlucose"] = glc_cat


    # ---------------------------
    # One-Hot Encoding (drop_first=True during training)
    # ---------------------------
    df = pd.get_dummies(
        df,
        columns=["NewBMI", "NewInsulinScore", "NewGlucose"],
        drop_first=True
    )

    # ---------------------------
    # Expected columns from training
    # (exact list you printed from the notebook)
    # ---------------------------
    expected_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",

        # BMI dummies
        "NewBMI_Normal",
        "NewBMI_Overweight",
        "NewBMI_Obesity 1",
        "NewBMI_Obesity 2",
        "NewBMI_Obesity 3",

        # Insulin dummy
        "NewInsulinScore_Normal",

        # Glucose dummies
        "NewGlucose_Normal",
        "NewGlucose_Prediabetic",
        "NewGlucose_High"
    ]

    # Guarantee all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df.reindex(columns=expected_cols)


    # ---------------------------
    # SCALE FULL FEATURE VECTOR
    # ---------------------------
    # scaler was trained on the FULL DataFrame (not numeric-only)
    df_scaled = scaler.transform(df)

    return df_scaled
