import joblib
import pandas as pd




def test_model_prediction_shape():
model = joblib.load('models/model.pkl')
data = pd.read_csv('data/processed/clean_data.csv')


X = data.drop('charges', axis=1)
preds = model.predict(X)


assert len(preds) == len(X)