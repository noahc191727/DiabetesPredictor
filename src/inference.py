from .utils import load_model, load_scaler
from .preprocess import preprocess_input


def predict(user_values):
    """
    Preprocess input and return prediction + probability.
    """
    model = load_model()
    scaler = load_scaler()
    X = preprocess_input(user_values, scaler)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return pred, prob
