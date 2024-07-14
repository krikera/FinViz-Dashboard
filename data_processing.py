import pandas as pd
import streamlit as st
from utils import identify_recurring_transactions

def prepare_data(financial_data):
    financial_data['Net Flow'] = financial_data['Deposits'] - financial_data['Withdrawls']
    financial_data['Date'] = pd.to_datetime(financial_data['Date'])
    return financial_data[['Date', 'Net Flow']]

def perform_eda(financial_data, chart_type):
    from visualization import visualize_data

    numeric_columns = ['Deposits', 'Withdrawls', 'Balance']
    for col in numeric_columns:
        if (col in financial_data.columns) and (financial_data[col].dtype == object):
            financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce')

    financial_data = identify_recurring_transactions(financial_data)

    summary_stats = financial_data.describe()
    st.write("Summary Statistics:")
    st.write(summary_stats)

    visualize_data(financial_data, 'Deposits', 'Deposits Over Time', chart_type)
    visualize_data(financial_data, 'Withdrawls', 'Withdrawals Over Time', chart_type)
    visualize_data(financial_data, 'Balance', 'Balance Over Time', chart_type)

def track_budget(data, budget_limits):
    data['Budget Limit'] = data['Category'].map(budget_limits)
    data['Total Withdrawls'] = data.groupby('Category')['Withdrawls'].transform('sum')
    data['Over Budget'] = data['Total Withdrawls'] > data['Budget Limit']
    return data