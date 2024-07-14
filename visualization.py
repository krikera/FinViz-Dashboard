import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from data_processing import track_budget

def visualize_data(data, column, title, chart_type):
    plt.figure(figsize=(10, 6))
    if chart_type == 'Line Chart':
        plt.plot(data['Date'], data[column], marker='o')
    elif chart_type == 'Bar Chart':
        plt.bar(data['Date'], data[column])
    elif chart_type == 'Histogram':
        plt.hist(data[column], bins=30)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    st.pyplot(plt)

def visualize_cash_flow(data, chart_type):
    data['Date'] = pd.to_datetime(data['Date'])
    data['YearMonth'] = data['Date'].dt.to_period('M')
    
    # Group by YearMonth and sum only the numeric columns
    cash_flow = data.groupby('YearMonth').agg({
        'Deposits': 'sum',
        'Withdrawls': 'sum'
    }).reset_index()
    
    cash_flow['Net Flow'] = cash_flow['Deposits'] - cash_flow['Withdrawls']
    cash_flow['Date'] = cash_flow['YearMonth'].dt.to_timestamp()
    
    visualize_data(cash_flow, 'Net Flow', 'Monthly Cash Flow', chart_type)

def visualize_budget_tracking(data, chart_type, budget_limits):
    data = track_budget(data, budget_limits)
    over_budget = data[data['Over Budget']]
    
    if over_budget.empty:
        st.write("No categories are over budget.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = over_budget['Category'].unique()
        actual_spending = over_budget.groupby('Category')['Total Withdrawls'].first()
        budget_limits = over_budget.groupby('Category')['Budget Limit'].first()

        x = range(len(categories))
        width = 0.35

        ax.bar(x, actual_spending, width, label='Actual Spending')
        ax.bar([i + width for i in x], budget_limits, width, label='Budget Limit')

        ax.set_ylabel('Amount')
        ax.set_title('Over Budget Categories')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(categories)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.write("Budget Tracking Details:")
    st.write(data[['Category', 'Total Withdrawls', 'Budget Limit', 'Over Budget']])

def visualize_spending_patterns_and_predictions(data, chart_type):
    from model import train_predict_model

    visualize_data(data, 'Withdrawls', 'Spending Patterns', chart_type)
    predictions = train_predict_model(data)
    visualize_data(predictions, 'Predicted', 'Spending Predictions', chart_type)