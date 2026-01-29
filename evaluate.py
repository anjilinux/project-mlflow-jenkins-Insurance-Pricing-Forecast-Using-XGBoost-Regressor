import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


model = joblib.load('models/model.pkl')
data = pd.read_csv('data/processed/clean_data.csv')


X = data.drop('charges', axis=1)
y = data['charges']


preds = model.predict(X)
rmse = mean_squared_error(y, preds, squared=False)


print(f"RMSE: {rmse}")