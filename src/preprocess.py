import pandas as pd


def preprocess_input(data: dict, scaler):
    """
    Preprocess a patient input dictionary using the same
    transformations as during training.
    """
    df = pd.DataFrame([data])

    # ---------------------------
    # Feature Engineering
    # ---------------------------
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

    ins = df.loc[0, "Insulin"]
    ins_cat = "Normal" if 16 <= ins <= 166 else "Abnormal"

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
    # One-hot encoding
    # ---------------------------
    df = pd.get_dummies(
        df,
        columns=["NewBMI", "NewInsulinScore", "NewGlucose"],
        drop_first=True,
    )

    expected_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "NewBMI_Normal",
        "NewBMI_Overweight",
        "NewBMI_Obesity 1",
        "NewBMI_Obesity 2",
        "NewBMI_Obesity 3",
        "NewInsulinScore_Normal",
        "NewGlucose_Normal",
        "NewGlucose_Prediabetic",
        "NewGlucose_High",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=expected_cols)

    df_scaled = scaler.transform(df)
    return df_scaled
