import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

df = pd.read_csv('data/raw/creditcard.csv')

print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum().sum()}") # Should be 0
print(f"Fraud ratio: {df['Class'].mean():.4%}")     # Should be ~0.17%

scaler = RobustScaler()
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

os.makedirs('data/processed', exist_ok=True)
df.to_parquet('data/processed/creditcard_clean.parquet', index=False)
print("Successfully saved to data/processed/creditcard_clean.parquet")