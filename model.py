import kagglehub
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download and load dataset
path = kagglehub.dataset_download("dsfelix/us-stores-sales")
df = pd.read_csv(path + '/sales.csv')

# Clean data
df['Inventory'] = df['Inventory'].clip(lower=0)

# Feature Engineering
df['sales_gap'] = df['Budget Sales'] - df['Sales']
df['profit_gap'] = df['Budget Profit'] - df['Profit']
df['inventory_efficiency'] = df['Profit'] / (df['Inventory'] + 1)
df['holding_cost'] = df['Inventory'] * 0.02
df['true_cost'] = df['COGS'] + df['Total Expenses'] + df['holding_cost']
df['true_margin'] = df['Sales'] - df['true_cost']

# Define backlog label (1 = backlog risk, 0 = healthy)
df['is_backlog'] = (
    (df['sales_gap'] > 0) &
    (df['Inventory'] > df['Inventory'].mean()) &
    (df['true_margin'] < df['true_margin'].mean())
).astype(int)

# Features for the model
features = [
    'Sales', 'COGS', 'Total Expenses', 'Inventory',
    'Budget Sales', 'sales_gap', 'profit_gap',
    'inventory_efficiency', 'holding_cost', 'true_margin'
]

X = df[features]
y = df['is_backlog']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(classification_report(y_test, y_pred))

# Feature importance
print("=" * 60)
print("MOST IMPORTANT FEATURES FOR BACKLOG PREDICTION")
print("=" * 60)
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.to_string())