import pandas as pd
from sklearn.preprocessing import StandardScaler




def preprocess_data(input_path, output_path):
df = pd.read_csv(input_path)


df = pd.get_dummies(df, drop_first=True)


scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children']
df[num_cols] = scaler.fit_transform(df[num_cols])


df.to_csv(output_path, index=False)


if __name__ == '__main__':
preprocess_data('data/raw/insurance.csv', 'data/processed/clean_data.csv')