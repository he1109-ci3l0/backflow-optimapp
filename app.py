from flask import Flask, render_template, jsonify
import kagglehub
import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def prepare_data():
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
    df['is_backlog'] = (
        (df['sales_gap'] > 0) &
        (df['Inventory'] > df['Inventory'].mean()) &
        (df['true_margin'] < df['true_margin'].mean())
    ).astype(int)
    df['recommended_discount'] = (
        (df['holding_cost'] / df['Sales']) * 100
    ).clip(0, 50).round(2)

    return df

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(df):
    features = [
        'Sales', 'COGS', 'Total Expenses', 'Inventory',
        'Budget Sales', 'sales_gap', 'profit_gap',
        'inventory_efficiency', 'holding_cost', 'true_margin'
    ]
    X = df[features]
    y = df['is_backlog']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, features

# Load data and train model on startup
df = prepare_data()
model, features = train_model(df)

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def summary():
    summary = df.groupby('Product').agg(
        avg_sales=('Sales', 'mean'),
        avg_inventory=('Inventory', 'mean'),
        total_profit=('Profit', 'sum'),
        avg_holding_cost=('holding_cost', 'mean'),
        avg_true_margin=('true_margin', 'mean'),
        avg_discount=('recommended_discount', 'mean'),
        backlog_count=('is_backlog', 'sum')
    ).round(2).reset_index()
    summary['risk_score'] = (summary['backlog_count'] / summary['backlog_count'].max() * 100).round(2)
    summary = summary.sort_values('risk_score', ascending=False)
    return jsonify(summary.to_dict(orient='records'))

@app.route('/api/high_risk')
def high_risk():
    high_risk = df[df['is_backlog'] == 1].groupby('Product').agg(
        avg_sales=('Sales', 'mean'),
        avg_inventory=('Inventory', 'mean'),
        total_profit=('Profit', 'sum'),
        avg_holding_cost=('holding_cost', 'mean'),
        avg_discount=('recommended_discount', 'mean')
    ).round(2).reset_index()
    return jsonify(high_risk.to_dict(orient='records'))

@app.route('/api/charts')
def charts():
    charts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts_data.json')
    with open(charts_path) as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
    