import pandas as pd
import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import basic utilities
try:
    from utils import custom_date_parser, categorize_transaction, enhanced_transaction_analysis, smart_search_transactions, generate_insights_summary, safe_csv_reader, normalize_column_names
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("Some advanced features may not be available in this deployment.")

try:
    from data_processing import prepare_data, perform_eda, track_budget
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

try:
    from visualization import visualize_cash_flow, visualize_budget_tracking, visualize_spending_patterns_and_predictions
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from model import detect_anomalies, train_predict_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Import enterprise features (optional)
try:
    from auth import get_auth, show_auth_sidebar, handle_auth_modals, require_auth, get_current_user_id
    from config import display_config_ui, get_config
    from database import get_database, FinancialTransaction
    from performance import get_performance_optimizer, show_performance_metrics, cached_data_processing
    from advanced_visualizations import get_visualization_engine
    ENTERPRISE_FEATURES = True
except ImportError as e:
    print(f"Enterprise features not available in deployment: {e}")
    ENTERPRISE_FEATURES = False

# Import enhanced features (optional)
try:
    from enhanced_export import create_enhanced_export_interface
    ENHANCED_EXPORT_AVAILABLE = True
except ImportError:
    ENHANCED_EXPORT_AVAILABLE = False

def main():
    st.set_page_config(
        page_title='FinViz Dashboard', 
        page_icon="💰", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply dark theme
    apply_dark_theme_styling()
    
    # Render header
    render_header()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your financial data file",
        type=['csv'],
        help="Upload a CSV file containing your financial transaction data"
    )
    
    if uploaded_file is not None:
        # Process file
        financial_data = process_uploaded_file(uploaded_file)
        if financial_data is not None:
            # Basic visualizations
            render_basic_visualizations(financial_data)
            
            # Data summary
            st.subheader("📊 Data Summary")
            st.write(financial_data.describe())
            
            # Basic export
            render_basic_export(financial_data)

def apply_dark_theme_styling():
    """Apply dark theme CSS styling"""
    dark_css = """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a67 100%);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #1e3a5f, #2d4a67);
        color: #fafafa;
        border: 1px solid #404040;
        border-radius: 8px;
    }
    
    .stSidebar > div {
        background-color: #1e1e1e;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

def render_header():
    """Render header"""
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">💰 FinViz Dashboard</h1>
        <p class="header-subtitle">Professional Financial Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    """Basic file processing"""
    try:
        # Read CSV
        financial_data = pd.read_csv(uploaded_file)
        
        # Basic data cleaning
        if 'Date' in financial_data.columns:
            financial_data['Date'] = pd.to_datetime(financial_data['Date'], errors='coerce')
            financial_data = financial_data.dropna(subset=['Date'])
        
        # Ensure required columns exist
        if 'Deposits' not in financial_data.columns:
            financial_data['Deposits'] = 0
        if 'Withdrawls' not in financial_data.columns:
            financial_data['Withdrawls'] = 0
            
        st.success(f"✅ Successfully loaded {len(financial_data)} transactions!")
        return financial_data
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def render_basic_visualizations(data):
    """Render basic visualizations"""
    st.subheader("📈 Financial Visualizations")
    
    if 'Date' in data.columns:
        # Monthly summary
        data['Month'] = data['Date'].dt.to_period('M')
        monthly_data = data.groupby('Month').agg({
            'Deposits': 'sum',
            'Withdrawls': 'sum'
        }).reset_index()
        
        if not monthly_data.empty:
            monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
            monthly_data['Net Flow'] = monthly_data['Deposits'] - monthly_data['Withdrawls']
            
            # Cash flow chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=monthly_data['Deposits'],
                name='Income',
                marker_color='#28a745'
            ))
            fig.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=-monthly_data['Withdrawls'],
                name='Expenses',
                marker_color='#dc3545'
            ))
            
            fig.update_layout(
                title="Monthly Cash Flow",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_income = data['Deposits'].sum()
        st.metric("Total Income", f"${total_income:,.2f}")
    with col2:
        total_expenses = data['Withdrawls'].sum()
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        net_flow = total_income - total_expenses
        st.metric("Net Flow", f"${net_flow:,.2f}")
    with col4:
        avg_transaction = data['Withdrawls'].mean()
        st.metric("Avg Transaction", f"${avg_transaction:.2f}")

def render_basic_export(data):
    """Basic export functionality"""
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download CSV"):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="financial_data.csv">Click to Download</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📊 Download JSON"):
            json_data = data.to_json(orient='records', date_format='iso')
            b64 = base64.b64encode(json_data.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="financial_data.json">Click to Download</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main() 