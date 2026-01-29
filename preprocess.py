import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    le = LabelEncoder()
    df["smoker"] = le.fit_transform(df["smoker"])
    df["region"] = le.fit_transform(df["region"])

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("insurance.csv",
                    "clean_data.csv")
