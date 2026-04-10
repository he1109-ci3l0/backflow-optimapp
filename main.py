import kagglehub
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("dsfelix/us-stores-sales")
print("Path to dataset:", path)

# Load the CSV
df = pd.read_csv(path + '/sales.csv')
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())
