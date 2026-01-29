import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


mlflow.set_experiment("Insurance_Pricing_XGBoost")


data = pd.read_csv('data/processed/clean_data.csv')
X = data.drop('charges', axis=1)
y = data['charges']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run():
model = xgb.XGBRegressor(
n_estimators=200,
max_depth=5,
learning_rate=0.1,
random_state=42
)


model.fit(X_train, y_train)


preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)


mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 5)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)


mlflow.xgboost.log_model(model, "model")


joblib.dump(model, "models/model.pkl")