import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

mlflow.set_experiment("Insurance_Pricing_XGBoost_v07")
mlflow.set_tracking_uri("file:/var/lib/jenkins/mlflow_clean")
df = pd.read_csv("clean_data.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05
    })

    mlflow.xgboost.log_model(model, "model")

    joblib.dump(model, "model.pkl")
