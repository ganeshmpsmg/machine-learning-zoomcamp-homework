import pandas as pd
import numpy as np

df = pd.read_csv('data/hour.csv')
print(f"Dataset shape: {df.shape}")
print("\nColumns and Data Types:")
print(df.dtypes)
print("\nMissing Values per column:")
print(df.isnull().sum())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nTarget 'cnt' Distribution Summary:")
print(df['cnt'].describe())
print(f"Skewness of cnt: {df['cnt'].skew():.4f}")
print(f"Kurtosis of cnt: {df['cnt'].kurt():.4f}")
print(f"Min date: {df['dteday'].min()}, Max date: {df['dteday'].max()}")
print("\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")
