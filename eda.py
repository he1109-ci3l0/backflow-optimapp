import kagglehub
import pandas as pd
import numpy as np

# Download and load dataset
path = kagglehub.dataset_download("dsfelix/us-stores-sales")
df = pd.read_csv(path + '/sales.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - US STORES SALES")
print("=" * 60)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Missing values:\n{df.isnull().sum()}")

# 2. Data Types
print("\n2. DATA TYPES")
print(df.dtypes)

# 3. Basic Statistics
print("\n3. BASIC STATISTICS")
print(df[['Sales', 'COGS', 'Profit', 'Inventory', 'Budget Sales']].describe())

# 4. Product Analysis
print("\n4. PRODUCT ANALYSIS")
print(f"   Unique products: {df['Product'].nunique()}")
print(f"   Products by type:\n{df['Product Type'].value_counts()}")

# 5. Sales vs Budget
print("\n5. SALES PERFORMANCE vs BUDGET")
total_sales = df['Sales'].sum()
total_budget = df['Budget Sales'].sum()
print(f"   Total Sales: ${total_sales:,.2f}")
print(f"   Total Budget Sales: ${total_budget:,.2f}")
print(f"   Variance: ${total_budget - total_sales:,.2f}")

# 6. Inventory Analysis
print("\n6. INVENTORY ANALYSIS")
print(f"   Average Inventory Value: ${df['Inventory'].mean():,.2f}")
print(f"   Max Inventory Value: ${df['Inventory'].max():,.2f}")
print(f"   Min Inventory Value: ${df['Inventory'].min():,.2f}")
# 7. Products losing money
print("\n7. PRODUCTS WITH NEGATIVE PROFIT")
losing = df[df['Profit'] < 0].groupby('Product')['Profit'].sum().sort_values()
print(losing)

# 8. Sales trend per product
print("\n8. AVERAGE SALES PER PRODUCT")
product_sales = df.groupby('Product')[['Sales', 'Budget Sales', 'Inventory', 'Profit']].mean().round(2)
print(product_sales)
