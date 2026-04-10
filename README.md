# Backflow Optimization App

## Overview
A data-driven supply chain management application designed to predict and prevent product backlog by analyzing inventory holding costs and recommending strategic discounts.

## Problem Statement
Companies often face significant losses from dead stock and accumulated inventory costs. This app identifies at-risk products before they become backlog and recommends optimal discount strategies to maintain profitability while clearing inventory.

## Features
- **Exploratory Data Analysis** — Comprehensive analysis of sales, inventory, and profit metrics
- **Risk Scoring** — Machine learning model predicting backlog risk for each product
- **Holding Cost Calculation** — True cost analysis including manufacturing, storage, and carrying costs
- **Discount Recommendations** — Strategic discount suggestions based on accumulated holding costs
- **Interactive Dashboard** — Real-time visualization of at-risk products and performance metrics
- **Plotly Charts** — Professional visualizations including sales vs budget, risk scores, and discount strategies

## Tech Stack
- **Backend:** Flask, Python
- **Data Analysis:** Pandas, Scikit-learn
- **Visualization:** Plotly
- **Deployment:** Render
- **Data Source:** Kaggle US Stores Sales Dataset

## How It Works
1. Loads real sales data from Kaggle
2. Engineers features including sales gaps, inventory efficiency, and true margins
3. Trains Random Forest model to identify backlog patterns
4. Calculates holding costs accumulating over time
5. Recommends discounts to recover costs before dead stock occurs
6. Displays insights in interactive web dashboard

## Getting Started
Visit: https://backflow-optimapp.onrender.com

## Author
Built with Python, data science, and supply chain optimization
Belem Cisneros Diaz
