import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

def load_model():
    """Load the trained XGBoost model."""
    return joblib.load(MODEL_PATH)

def load_scaler():
    """Load the StandardScaler used in training."""
    return joblib.load(SCALER_PATH)
