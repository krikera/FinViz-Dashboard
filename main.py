import pandas as pd
import streamlit as st
import base64
from utils import custom_date_parser, categorize_transaction
from data_processing import prepare_data, perform_eda, track_budget
from visualization import visualize_cash_flow, visualize_budget_tracking, visualize_spending_patterns_and_predictions
from model import detect_anomalies, train_predict_model

def main():
    st.set_page_config(page_title='Finance Analytics', page_icon="💰", layout="wide")
    st.title('Your Finance Dashboard')

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            financial_data = pd.read_csv(uploaded_file, encoding='latin1')
            financial_data['Date'] = pd.to_datetime(financial_data['Date'].apply(custom_date_parser))
            financial_data['Deposits'] = pd.to_numeric(financial_data['Deposits'].str.replace(',', ''), errors='coerce')
            financial_data['Withdrawls'] = pd.to_numeric(financial_data['Withdrawls'].str.replace(',', ''), errors='coerce')
            financial_data['Category'] = financial_data['Description'].apply(categorize_transaction)
        except Exception as e:
            st.error(f"Error: {e}. Could not read the CSV file.")
            return

        # Detect anomalies
        financial_data = detect_anomalies(financial_data)
        st.write("Detected Anomalies:")
        st.write(financial_data[financial_data['Anomaly'] == -1])

        # Budget Settings
        st.sidebar.header('Budget Settings')
        budget_limits = {}
        for category in financial_data['Category'].unique():
            budget_limits[category] = st.sidebar.number_input(f'Budget for {category}', min_value=0.0, value=0.0, key=f'budget_{category}')

        # Chart Settings
        st.sidebar.header('Chart Settings')
        chart_type = st.sidebar.selectbox('Select Chart Type', ['Line Chart', 'Bar Chart', 'Histogram'])

        perform_eda(financial_data, chart_type)

        # Filters and Settings
        st.sidebar.header('Filters and Settings')
        categories = st.sidebar.multiselect('Select Categories', financial_data['Category'].unique())
        min_deposit = st.sidebar.number_input('Minimum Deposit Amount', min_value=0.0, value=0.0)
        max_deposit = st.sidebar.number_input('Maximum Deposit Amount', min_value=0.0, value=financial_data['Deposits'].max())
        min_withdrawal = st.sidebar.number_input('Minimum Withdrawal Amount', min_value=0.0, value=0.0)
        max_withdrawal = st.sidebar.number_input('Maximum Withdrawal Amount', min_value=0.0, value=financial_data['Withdrawls'].max())
        search_keywords = st.sidebar.text_input('Search Keywords in Description')

        min_date = financial_data['Date'].min().date()
        max_date = financial_data['Date'].max().date()
        start_date = st.sidebar.date_input('Start Date', min_date)
        end_date = st.sidebar.date_input('End Date', max_date)

        filtered_data = financial_data[
            (financial_data['Category'].isin(categories)) &
            (financial_data['Date'] >= pd.to_datetime(start_date)) &
            (financial_data['Date'] <= pd.to_datetime(end_date)) &
            (financial_data['Deposits'] >= min_deposit) &
            (financial_data['Deposits'] <= max_deposit) &
            (financial_data['Withdrawls'] >= min_withdrawal) &
            (financial_data['Withdrawls'] <= max_withdrawal)
        ]

        if search_keywords:
            filtered_data = filtered_data[filtered_data['Description'].str.contains(search_keywords, case=False)]

        st.sidebar.subheader('Summary Statistics')
        st.sidebar.write(filtered_data.describe())

        # Customize Dashboard
        st.sidebar.header('Customize Dashboard')
        available_widgets = {
            "Cash Flow": visualize_cash_flow,
            "Budget Tracking": lambda data, chart: visualize_budget_tracking(data, chart, budget_limits),
            "Spending Patterns and Predictions": visualize_spending_patterns_and_predictions
        }

        selected_widgets = st.sidebar.multiselect('Select Widgets', list(available_widgets.keys()))

        for widget in selected_widgets:
            st.subheader(widget)
            if widget in available_widgets:
                available_widgets[widget](filtered_data, chart_type)
            else:
                st.warning(f"Widget '{widget}' not available.")

        # Data Entry
        st.sidebar.header('Data Entry')
        st.sidebar.subheader('Add New Transaction')
        new_date = st.sidebar.date_input('Date', value=pd.to_datetime('today'))
        new_description = st.sidebar.text_input('Description')
        new_deposit = st.sidebar.number_input('Deposit Amount', min_value=0.0, value=0.0)
        new_withdrawal = st.sidebar.number_input('Withdrawal Amount', min_value=0.0, value=0.0)
        new_tags = st.sidebar.text_input('Tags (comma-separated)')

        if st.sidebar.button('Add Transaction'):
            new_transaction = pd.DataFrame({
                'Date': [new_date],
                'Description': [new_description],
                'Deposits': [new_deposit],
                'Withdrawls': [new_withdrawal],
                'Balance': [filtered_data['Balance'].iloc[-1] + new_deposit - new_withdrawal],
                'Category': [categorize_transaction(new_description)],
                'Recurring': [False],
                'Tags': [new_tags]
            })
            filtered_data = pd.concat([filtered_data, new_transaction], ignore_index=True)
            filtered_data.to_csv('finance.csv', index=False)
            st.success('Transaction added successfully.')

        # Export Data
        st.sidebar.markdown("---")
        if st.sidebar.button("Export Filtered Data"):
            export_filename = "filtered_data.csv"
            with st.spinner(f"Exporting data to {export_filename}..."):
                csv = filtered_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success(f"Data exported to {export_filename}.")

if __name__ == '__main__':
    main()