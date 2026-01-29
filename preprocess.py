import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    categorical_cols = ["sex", "smoker", "region"]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "insurance.csv",
        "clean_data.csv"
    )
