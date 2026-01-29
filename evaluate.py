import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("clean_data.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

model = joblib.load("model.pkl")
preds = model.predict(X)

print("MSE:", mean_squared_error(y, preds))
print("R2 Score:", r2_score(y, preds))
