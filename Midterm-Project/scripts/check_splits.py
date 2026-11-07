import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/hour.csv')

print(f"Total rows: {len(df)}")
print(f"Date range: {df['dteday'].min()} to {df['dteday'].max()}")

# Verify leak columns exist
assert 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns
assert (df['casual'] + df['registered'] == df['cnt']).all()
print("Verified: casual + registered == cnt (strict leakage confirmed).")

# Check time sorting
df['dteday'] = pd.to_datetime(df['dteday'])
is_sorted = df['dteday'].is_monotonic_increasing
print(f"Is dteday monotonic increasing: {is_sorted}")

# Split strategy:
# Chronological 70% train, 15% val, 15% test
n = len(df)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
n_test = n - n_train - n_val

train_df = df.iloc[:n_train].copy()
val_df = df.iloc[n_train:n_train + n_val].copy()
test_df = df.iloc[n_train + n_val:].copy()

print(f"Train set: {len(train_df)} rows ({train_df['dteday'].min().strftime('%Y-%m-%d')} to {train_df['dteday'].max().strftime('%Y-%m-%d')})")
print(f"Val set:   {len(val_df)} rows ({val_df['dteday'].min().strftime('%Y-%m-%d')} to {val_df['dteday'].max().strftime('%Y-%m-%d')})")
print(f"Test set:  {len(test_df)} rows ({test_df['dteday'].min().strftime('%Y-%m-%d')} to {test_df['dteday'].max().strftime('%Y-%m-%d')})")
