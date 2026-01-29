import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("artifacts/model.pkl")

    sample = np.array([[40, 30.0, 2, 0, 1]])
    prediction = model.predict(sample)

    assert prediction.shape[0] == 1
    assert prediction[0] > 0
