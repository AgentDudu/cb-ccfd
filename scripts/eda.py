import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw/creditcard.csv')

print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum().sum()}") # Should be 0
print(f"Fraud ratio: {df['Class'].mean():.4%}")     # Should be ~0.17%

feature_cols = [col for col in df.columns if col != "Class"]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

os.makedirs('data/processed', exist_ok=True)
df.to_parquet('data/processed/creditcard_clean.parquet', index=False)
print("Successfully saved to data/processed/creditcard_clean.parquet")
