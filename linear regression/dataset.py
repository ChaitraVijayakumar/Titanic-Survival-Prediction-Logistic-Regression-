import pandas as pd

# Load the Kaggle dataset
df = pd.read_csv('linear regression/dataset.csv')        # Change name if your file is named differently

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())