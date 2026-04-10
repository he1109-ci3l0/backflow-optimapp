import kagglehub
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

# Download and load dataset
path = kagglehub.dataset_download("dsfelix/us-stores-sales")
df = pd.read_csv(path + '/sales.csv')

# Clean data
df['Inventory'] = df['Inventory'].clip(lower=0)

# Feature Engineering
df['sales_gap'] = df['Budget Sales'] - df['Sales']
df['profit_gap'] = df['Budget Profit'] - df['Profit']
df['holding_cost'] = df['Inventory'] * 0.02
df['true_cost'] = df['COGS'] + df['Total Expenses'] + df['holding_cost']
df['true_margin'] = df['Sales'] - df['true_cost']
df['recommended_discount'] = (
    (df['holding_cost'] / df['Sales']) * 100
).clip(0, 50).round(2)

# Aggregate by product
product_summary = df.groupby('Product').agg(
    avg_sales=('Sales', 'mean'),
    avg_budget_sales=('Budget Sales', 'mean'),
    avg_inventory=('Inventory', 'mean'),
    avg_holding_cost=('holding_cost', 'mean'),
    total_profit=('Profit', 'sum'),
    avg_true_margin=('true_margin', 'mean'),
    avg_discount=('recommended_discount', 'mean')
).round(2).reset_index()

# Risk score
product_summary['risk_score'] = (
    (product_summary['avg_holding_cost'] / product_summary['avg_sales'] * 100)
).round(2)

# ============================================================
# CHART 1: RISK SCORE BY PRODUCT
# ============================================================
fig1 = px.bar(
    product_summary.sort_values('risk_score', ascending=False),
    x='Product',
    y='risk_score',
    title='Backlog Risk Score by Product',
    color='risk_score',
    color_continuous_scale='Reds',
    labels={'risk_score': 'Risk Score', 'Product': 'Product'}
)
fig1.update_layout(height=400)
chart1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')

# ============================================================
# CHART 2: SALES VS BUDGET
# ============================================================
fig2 = go.Figure(data=[
    go.Bar(
        name='Actual Sales',
        x=product_summary['Product'],
        y=product_summary['avg_sales'],
        marker_color='#4ecca3'
    ),
    go.Bar(
        name='Budget Sales',
        x=product_summary['Product'],
        y=product_summary['avg_budget_sales'],
        marker_color='#e94560'
    )
])
fig2.update_layout(
    title='Actual Sales vs Budget Sales by Product',
    barmode='group',
    height=400,
    xaxis_title='Product',
    yaxis_title='Sales ($)'
)
chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# CHART 3: HOLDING COST BY PRODUCT
# ============================================================
fig3 = px.bar(
    product_summary.sort_values('avg_holding_cost', ascending=False),
    x='Product',
    y='avg_holding_cost',
    title='Average Holding Cost by Product',
    color='avg_holding_cost',
    color_continuous_scale='YlOrRd',
    labels={'avg_holding_cost': 'Holding Cost ($)'}
)
fig3.update_layout(height=400)
chart3 = fig3.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# CHART 4: RECOMMENDED DISCOUNT BY PRODUCT
# ============================================================
fig4 = px.bar(
    product_summary.sort_values('avg_discount', ascending=False),
    x='Product',
    y='avg_discount',
    title='Recommended Discount Strategy by Product',
    color='avg_discount',
    color_continuous_scale='Blues',
    labels={'avg_discount': 'Discount (%)'}
)
fig4.update_layout(height=400)
chart4 = fig4.to_html(full_html=False, include_plotlyjs=False)

# ============================================================
# CHART 5: TRUE MARGIN VS PROFIT
# ============================================================
fig5 = go.Figure(data=[
    go.Bar(
        name='Total Profit',
        x=product_summary['Product'],
        y=product_summary['total_profit'],
        marker_color='#4ecca3'
    ),
    go.Bar(
        name='True Margin (after holding costs)',
        x=product_summary['Product'],
        y=product_summary['avg_true_margin'],
        marker_color='#e94560'
    )
])
fig5.update_layout(
    title='Total Profit vs True Margin After Holding Costs',
    barmode='group',
    height=400,
    xaxis_title='Product',
    yaxis_title='($)'
)
chart5 = fig5.to_html(full_html=False, include_plotlyjs=False)

# Save all charts
charts = {
    'chart1': chart1,
    'chart2': chart2,
    'chart3': chart3,
    'chart4': chart4,
    'chart5': chart5
}

with open('charts_data.json', 'w') as f:
    json.dump(charts, f)

print("=" * 60)
print("ALL CHARTS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("chart1: Risk Score by Product")
print("chart2: Sales vs Budget")
print("chart3: Holding Cost by Product")
print("chart4: Recommended Discounts")
print("chart5: Profit vs True Margin")
