from src.utils import load_model, load_scaler


def test_load_models():
    model = load_model()
    assert model is not None


def test_load_scalers():
    scaler = load_scaler()
    assert scaler is not None
