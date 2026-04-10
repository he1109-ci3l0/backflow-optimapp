import kagglehub
import pandas as pd
import numpy as np

# Download and load dataset
path = kagglehub.dataset_download("dsfelix/us-stores-sales")
df = pd.read_csv(path + '/sales.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Clean negative inventory values
df['Inventory'] = df['Inventory'].clip(lower=0)

# ============================================================
# FEATURE ENGINEERING - BACKLOG RISK INDICATORS
# ============================================================

# 1. Sales performance gap (positive = underperforming)
df['sales_gap'] = df['Budget Sales'] - df['Sales']
df['profit_gap'] = df['Budget Profit'] - df['Profit']

# 2. Sales efficiency (how much profit per dollar of inventory)
df['inventory_efficiency'] = df['Profit'] / (df['Inventory'] + 1)

# 3. Holding cost (2% of inventory value per period)
df['holding_cost'] = df['Inventory'] * 0.02

# 4. True product cost (COGS + Total Expenses + Holding Cost)
df['true_cost'] = df['COGS'] + df['Total Expenses'] + df['holding_cost']

# 5. True margin (Sales - True Cost)
df['true_margin'] = df['Sales'] - df['true_cost']

# 6. Recommended discount to recover holding costs
df['recommended_discount_pct'] = (
    (df['holding_cost'] / df['Sales']) * 100
).clip(0, 50).round(2)

# ============================================================
# BACKLOG RISK SCORE (0-100)
# ============================================================

# Higher score = higher risk
df['risk_score'] = (
    (df['sales_gap'] > 0).astype(int) * 30 +      # underperforming sales
    (df['profit_gap'] > 0).astype(int) * 30 +      # underperforming profit
    (df['Inventory'] > df['Inventory'].mean()).astype(int) * 20 +  # high inventory
    (df['true_margin'] < 0).astype(int) * 20       # negative true margin
)

# ============================================================
# SUMMARY BY PRODUCT
# ============================================================

summary = df.groupby('Product').agg(
    avg_sales=('Sales', 'mean'),
    avg_inventory=('Inventory', 'mean'),
    total_profit=('Profit', 'sum'),
    avg_holding_cost=('holding_cost', 'mean'),
    avg_true_margin=('true_margin', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_discount=('recommended_discount_pct', 'mean')
).round(2)

summary = summary.sort_values('avg_risk_score', ascending=False)

print("=" * 60)
print("BACKLOG RISK ANALYSIS BY PRODUCT")
print("=" * 60)
print(summary.to_string())

print("\n=== HIGH RISK PRODUCTS (score >= 50) ===")
high_risk = summary[summary['avg_risk_score'] >= 50]
print(high_risk[['avg_sales', 'avg_inventory', 'total_profit', 'avg_risk_score', 'avg_discount']].to_string())
