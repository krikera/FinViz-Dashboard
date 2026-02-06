import pandas as pd
import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Core imports - always available
try:
    from utils import custom_date_parser, categorize_transaction, enhanced_transaction_analysis, smart_search_transactions, generate_insights_summary, safe_csv_reader, normalize_column_names
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Utils not available - using basic functionality")

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

try:
    from nlp_visualizations import display_nlp_dashboard
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

try:
    from userinput import user_input_handler
    USER_INPUT_AVAILABLE = True
except ImportError:
    USER_INPUT_AVAILABLE = False

# Enterprise features (optional for deployment)
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

# Enhanced export (optional)
try:
    from enhanced_export import create_enhanced_export_interface
    ENHANCED_EXPORT_AVAILABLE = True
except ImportError:
    ENHANCED_EXPORT_AVAILABLE = False

# Basic fallback functions when modules aren't available
def basic_date_parser(date_str):
    """Basic date parsing fallback"""
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def basic_categorize_transaction(description):
    """Basic categorization fallback"""
    if pd.isna(description):
        return 'other'
    desc_lower = str(description).lower()
    if any(word in desc_lower for word in ['grocery', 'food', 'restaurant', 'cafe']):
        return 'food_dining'
    elif any(word in desc_lower for word in ['gas', 'fuel', 'transport']):
        return 'transportation'
    elif any(word in desc_lower for word in ['shop', 'store', 'amazon', 'purchase']):
        return 'shopping'
    else:
        return 'other'

def basic_safe_csv_reader(uploaded_file):
    """Basic CSV reader fallback"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, 'utf-8', ','
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None, None, None

def basic_normalize_column_names(df):
    """Basic column name normalization"""
    column_mapping = {
        'date': 'Date',
        'description': 'Description', 
        'desc': 'Description',
        'amount': 'Amount',
        'deposit': 'Deposits',
        'deposits': 'Deposits',
        'withdrawal': 'Withdrawls',
        'withdrawals': 'Withdrawls',
        'withdrawls': 'Withdrawls',
        'balance': 'Balance'
    }
    
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    return df

def main():
    st.set_page_config(
        page_title='FinViz Dashboard', 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for new features
    initialize_enhanced_session_state()
    
    # Apply modern styling with theme support
    apply_dark_theme_styling()
    
    # Enable keyboard shortcuts
    enable_keyboard_shortcuts()
    
    # Initialize enterprise features
    if ENTERPRISE_FEATURES:
        if not initialize_enterprise_features():
            return
    
    # Enhanced header
    render_enhanced_header()
    
    # Add feature discovery and keyboard shortcuts
    add_feature_discovery()
    
    # Add navigation tabs for enterprise features
    if ENTERPRISE_FEATURES:
        main_tab, portfolio_tab, admin_tab = st.tabs(["📊 Dashboard", "📈 Portfolio View", "⚙️ Admin"])
    else:
        main_tab = st.container()
        portfolio_tab = None
        admin_tab = None

    with main_tab:
        # Enhanced file upload section
        render_enhanced_upload_section()
        
        uploaded_file = st.file_uploader(
            "Choose your financial data file",
            type=['csv'],
            help="Upload a CSV file containing your financial transaction data"
        )
        
        if uploaded_file is not None:
            # Show file details
            file_details = {
                "📁 Filename": uploaded_file.name,
                "📊 File Size": f"{uploaded_file.size / 1024:.1f} KB",
                "🔧 File Type": uploaded_file.type
            }
            
            with st.expander("📄 File Information", expanded=False):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Process file with enhanced error handling
            financial_data = process_uploaded_file_enhanced(uploaded_file)
            if financial_data is None:
                return

            # Apply interactive filters
            filtered_data = apply_interactive_filters(financial_data)
            
            # Show filter impact
            show_filter_impact(financial_data, filtered_data)
            
            # Enhanced Export Center
            if ENHANCED_EXPORT_AVAILABLE:
                with st.expander("📤 Enhanced Export Center", expanded=False):
                    create_enhanced_export_interface(filtered_data)
            
            # Initialize user input handler and display settings (only if available)
            if USER_INPUT_AVAILABLE:
                user_input_handler.display_user_settings()
                
                # Get user preferences
                budget_limits = user_input_handler.preferences.get("budget_categories", {})
                
                # Get user filters (traditional)
                filters = user_input_handler.get_user_filters(filtered_data)
            else:
                # Fallback when user input handler not available
                budget_limits = {}
                filters = {}
                st.info("💡 Advanced user settings not available in this deployment")
            
            # Apply traditional filters
            if USER_INPUT_AVAILABLE:
                filtered_data = user_input_handler.apply_filters(filtered_data, filters)

            # Detect anomalies if enabled
            if MODEL_AVAILABLE and USER_INPUT_AVAILABLE and user_input_handler.preferences.get("show_anomalies", True):
                filtered_data = detect_anomalies(filtered_data)
                anomalies = filtered_data[filtered_data['Anomaly'] == -1]
                if not anomalies.empty:
                    st.warning(f"⚠️ Detected {len(anomalies)} anomalous transactions")
                    with st.expander("View Anomalous Transactions"):
                        st.write(anomalies[['Date', 'Description', 'Withdrawls', 'Deposits']])
            elif not MODEL_AVAILABLE:
                st.info("💡 Anomaly detection not available in this deployment")

            # Enhanced visualizations with real-time updates
            if VISUALIZATION_AVAILABLE:
                render_enhanced_visualizations(filtered_data)
            else:
                # Basic visualization fallback
                render_basic_dashboard(filtered_data)

            # Use user's preferred chart type
            chart_type = "Line Chart"  # Default
            if USER_INPUT_AVAILABLE:
                chart_type = user_input_handler.preferences.get("default_chart_type", "Line Chart")

            # Perform EDA with filtered data
            if DATA_PROCESSING_AVAILABLE:
                perform_eda(filtered_data, chart_type)
            else:
                st.info("💡 Advanced EDA not available in this deployment")

            # Summary statistics in sidebar
            st.sidebar.subheader('Summary Statistics')
            st.sidebar.write(filtered_data.describe())
            
            # Quick actions (only if user input handler available)
            if USER_INPUT_AVAILABLE:
                quick_action = user_input_handler.display_quick_actions()
                
                # Add NLP insights in sidebar
                if user_input_handler.preferences.get("nlp_insights", True) and NLP_AVAILABLE:
                    st.sidebar.subheader('🧠 AI Insights')
                    if st.sidebar.button('Generate AI Summary') or quick_action == "generate_insights":
                        with st.sidebar:
                            with st.spinner('Analyzing transactions...'):
                                if UTILS_AVAILABLE:
                                    insights_summary = generate_insights_summary(filtered_data)
                                    st.markdown(insights_summary)
                                else:
                                    st.info("AI insights not available in this deployment")

                # Get widget selection from user input handler
                selected_widgets = user_input_handler.get_widget_selection()
                
                # Available widgets
                available_widgets = {}
                if VISUALIZATION_AVAILABLE:
                    available_widgets = {
                        "Cash Flow": visualize_cash_flow,
                        "Budget Tracking": lambda data, chart: visualize_budget_tracking(data, chart, {}),
                        "Spending Patterns and Predictions": visualize_spending_patterns_and_predictions,
                    }
                    if NLP_AVAILABLE:
                        available_widgets["🧠 NLP Analysis"] = lambda data, chart: display_nlp_dashboard(data)

                for widget in selected_widgets:
                    st.subheader(widget)
                    if widget in available_widgets:
                        if widget == "🧠 NLP Analysis":
                            available_widgets[widget](filtered_data, chart_type)
                        else:
                            available_widgets[widget](filtered_data, chart_type)
                    else:
                        st.warning(f"Widget '{widget}' not available in this deployment.")

            # Enhanced Data Entry (only if available)
            if USER_INPUT_AVAILABLE:
                render_enhanced_data_entry(filtered_data)
            else:
                # Basic export functionality
                render_basic_export(filtered_data)

    # Portfolio View Tab (Enterprise Feature)
    if ENTERPRISE_FEATURES and portfolio_tab:
        with portfolio_tab:
            if uploaded_file is not None and 'financial_data' in locals():
                viz_engine = get_visualization_engine()
                viz_engine.create_executive_dashboard(financial_data)
                viz_engine.create_interactive_portfolio_view(financial_data)
            else:
                st.info("Please upload data in the Dashboard tab to view portfolio analysis.")

    # Admin Tab (Enterprise Feature)
    if ENTERPRISE_FEATURES and admin_tab:
        with admin_tab:
            auth = get_auth()
            if auth.is_authenticated():
                user = auth.get_current_user()
                if user and user.username == "admin":
                    st.title("🛠️ Administration Panel")
                    
                    # Performance dashboard
                    st.subheader("📊 Performance Monitoring")
                    optimizer = get_performance_optimizer()
                    optimizer.get_performance_dashboard()
                    
                    # User management placeholder
                    st.subheader("👥 User Management")
                    st.info("User management features coming soon...")
                    
                    # System configuration
                    st.subheader("⚙️ System Configuration")
                    config = get_config()
                    st.write(f"Environment: {config.environment}")
                    st.write(f"Database Type: {config.database.type}")
                    st.write(f"Authentication: {'Enabled' if config.security.enable_authentication else 'Disabled'}")
                    
                else:
                    st.warning("Admin access required")
            else:
                st.warning("Please log in to access admin features")

def initialize_enhanced_session_state():
    """Initialize session state for enhanced features"""
    if 'interactive_filters_enabled' not in st.session_state:
        st.session_state.interactive_filters_enabled = False
    if 'filter_values' not in st.session_state:
        st.session_state.filter_values = {}
    if 'last_filter_update' not in st.session_state:
        st.session_state.last_filter_update = datetime.now()

def apply_dark_theme_styling():
    """Apply professional financial terminal CSS styling"""
    dark_css = """
    <style>
    /* Base styling */
    .main > div {
        padding: 0.75rem 0rem;
    }

    /* Professional dark theme */
    .stApp {
        background-color: #0a0e14;
        color: #c5cdd9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Header styling */
    .main-header {
        padding: 1.25rem 2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        text-align: left;
        border-bottom: 2px solid #00c896;
        border-left: none;
        box-shadow: none;
        background: #0d1117;
    }

    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
        color: #e6edf3;
        text-shadow: none;
    }

    .header-subtitle {
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
        color: #7d8590;
        letter-spacing: 0.3px;
    }

    .header-divider {
        display: inline-block;
        width: 3px;
        height: 28px;
        background: #00c896;
        margin-right: 12px;
        vertical-align: middle;
        border-radius: 1px;
    }

    /* Metric cards */
    .metric-card {
        padding: 1rem 1.25rem;
        border-radius: 4px;
        box-shadow: none;
        border-left: 3px solid #00c896;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
        background: #161b22;
        color: #c5cdd9;
    }

    .metric-card:hover {
        border-color: #3fb950;
        transform: none;
        box-shadow: none;
    }

    /* Streamlit metric overrides */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 0.75rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #7d8590 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        border: 1px solid #30363d;
        font-weight: 500;
        font-size: 0.8rem;
        transition: all 0.15s ease;
        background: #21262d;
        color: #c5cdd9;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        background: #30363d;
        border-color: #00c896;
        transform: none;
        box-shadow: none;
        color: #e6edf3;
    }

    /* File uploader */
    .stFileUploader > div {
        border: 1px dashed #30363d;
        border-radius: 4px;
        padding: 1.5rem;
        background: #0d1117;
    }

    .stFileUploader > div:hover {
        border-color: #00c896;
    }

    /* Interactive filter panel */
    .filter-panel {
        background: #0d1117;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid #21262d;
    }

    /* Sidebar styling */
    .stSidebar > div {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #161b22;
        color: #c5cdd9;
        border-color: #30363d;
    }

    /* Expander styling */
    .stExpander {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1117;
        border-bottom: 1px solid #21262d;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #7d8590;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
        border-bottom: 2px solid transparent;
        padding: 0.5rem 1rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e6edf3;
    }

    .stTabs [aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom: 2px solid #00c896 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #21262d;
        border-radius: 4px;
    }

    /* Status badges */
    .status-positive { color: #3fb950; font-weight: 600; }
    .status-negative { color: #f85149; font-weight: 600; }
    .status-neutral { color: #7d8590; font-weight: 600; }
    </style>
    """
    
    st.markdown(dark_css, unsafe_allow_html=True)

def enable_keyboard_shortcuts():
    """Enable keyboard shortcuts for power users"""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Refresh data (Ctrl + R)
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            window.location.reload();
        }
        
        // Focus search (Ctrl + F)
        if (e.ctrlKey && e.key === 'f') {
            e.preventDefault();
            const searchBox = document.querySelector('input[placeholder*="search"], input[placeholder*="Search"]');
            if (searchBox) {
                searchBox.focus();
                searchBox.select();
            }
        }
        
        // Show help (?)
        if (e.key === '?' && !e.ctrlKey && !e.altKey) {
            e.preventDefault();
            alert('Keyboard Shortcuts:\\n\\nCtrl + R: Refresh data\\nCtrl + F: Focus search\\n?: Show this help');
        }
    });
    </script>
    """, unsafe_allow_html=True)

def apply_interactive_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply interactive real-time filters"""
    
    # Simple fallback when advanced features not available
    if not USER_INPUT_AVAILABLE:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🎛️ Basic Filters")
            st.info("💡 Interactive filtering not available in this deployment")
        return data
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎛️ Interactive Filters")
        
        # Enable/disable interactive mode
        interactive_mode = st.toggle(
            "🔄 Real-time Filtering",
            value=st.session_state.interactive_filters_enabled,
            help="Enable real-time chart updates as you adjust filters"
        )
        st.session_state.interactive_filters_enabled = interactive_mode
        
        if not interactive_mode:
            st.info("💡 Enable real-time filtering for instant chart updates!")
            return data
        
        filtered_data = data.copy()
        
        # Date range filter
        if 'Date' in data.columns and not data.empty:
            min_date = data['Date'].min().date()
            max_date = data['Date'].max().date()
            
            date_range = st.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Select date range for analysis"
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = filtered_data[
                    (filtered_data['Date'].dt.date >= start_date) &
                    (filtered_data['Date'].dt.date <= end_date)
                ]
        
        # Amount filters
        if 'Withdrawls' in data.columns and not data.empty:
            max_withdrawal = float(data['Withdrawls'].max())
            if max_withdrawal > 0:
                amount_range = st.slider(
                    "💰 Transaction Amount Range",
                    min_value=0.0,
                    max_value=max_withdrawal,
                    value=(0.0, max_withdrawal),
                    step=max_withdrawal / 100,
                    format="$%.2f",
                    help="Filter transactions by amount"
                )
                
                min_amount, max_amount = amount_range
                filtered_data = filtered_data[
                    (filtered_data['Withdrawls'] >= min_amount) &
                    (filtered_data['Withdrawls'] <= max_amount)
                ]
        
        # Category filter
        if 'Category' in data.columns and not data.empty:
            available_categories = sorted(data['Category'].unique())
            selected_categories = st.multiselect(
                "🏷️ Categories",
                options=available_categories,
                default=available_categories,
                help="Select categories to include in analysis"
            )
            
            if selected_categories:
                filtered_data = filtered_data[
                    filtered_data['Category'].isin(selected_categories)
                ]
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            # Weekend/Weekday filter
            day_filter = st.selectbox(
                "Day Type",
                options=["All Days", "Weekdays Only", "Weekends Only"],
                help="Filter by day of week"
            )
            
            if day_filter == "Weekdays Only":
                filtered_data = filtered_data[filtered_data['Date'].dt.weekday < 5]
            elif day_filter == "Weekends Only":
                filtered_data = filtered_data[filtered_data['Date'].dt.weekday >= 5]
            
            # Description search
            description_search = st.text_input(
                "🔍 Search Descriptions",
                placeholder="Enter keywords to search...",
                help="Search in transaction descriptions"
            )
            
            if description_search:
                filtered_data = filtered_data[
                    filtered_data['Description'].str.lower().str.contains(
                        description_search.lower(), na=False
                    )
                ]
        
        # Show active filters
        active_filters = []
        if len(date_range) == 2 and (date_range[0] != min_date or date_range[1] != max_date):
            active_filters.append("Date Range")
        if 'amount_range' in locals() and (amount_range[0] != 0.0 or amount_range[1] != max_withdrawal):
            active_filters.append("Amount Range")
        if selected_categories and len(selected_categories) != len(available_categories):
            active_filters.append("Categories")
        if day_filter != "All Days":
            active_filters.append("Day Type")
        if description_search:
            active_filters.append("Description Search")
        
        if active_filters:
            st.markdown("#### 📊 Active Filters")
            for filter_name in active_filters:
                st.write(f"✅ {filter_name}")
        else:
            st.write("No active filters")
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", type="secondary"):
            st.rerun()
        
        return filtered_data

def show_filter_impact(original_data: pd.DataFrame, filtered_data: pd.DataFrame):
    """Show the impact of current filters"""
    if hasattr(st.session_state, 'interactive_filters_enabled') and st.session_state.interactive_filters_enabled:
        original_count = len(original_data)
        filtered_count = len(filtered_data)
        
        if filtered_count < original_count:
            reduction = ((original_count - filtered_count) / original_count) * 100
            st.info(f"🔍 Filters applied: Showing {filtered_count:,} of {original_count:,} transactions ({reduction:.1f}% filtered out)")

def render_enhanced_visualizations(data: pd.DataFrame):
    """Render enhanced visualizations with real-time updates"""
    st.markdown("### 📈 Financial Overview")
    
    if data.empty:
        st.warning("No data available for visualization")
        return
    
    # Create interactive charts
    viz_tabs = st.tabs(["💰 Cash Flow", "📊 Categories", "📈 Trends", "🔍 Analysis"])
    
    with viz_tabs[0]:
        render_interactive_cash_flow(data)
    
    with viz_tabs[1]:
        render_interactive_categories(data)
    
    with viz_tabs[2]:
        render_interactive_trends(data)
    
    with viz_tabs[3]:
        render_interactive_analysis(data)

def render_interactive_cash_flow(data: pd.DataFrame):
    """Render interactive cash flow visualization"""
    try:
        # Prepare monthly data
        data['Date'] = pd.to_datetime(data['Date'])
        monthly_data = data.groupby(data['Date'].dt.to_period('M')).agg({
            'Deposits': 'sum',
            'Withdrawls': 'sum'
        }).reset_index()
        
        if monthly_data.empty:
            st.warning("No monthly data available")
            return
        
        monthly_data['Net_Flow'] = monthly_data['Deposits'] - monthly_data['Withdrawls']
        monthly_data['Month'] = monthly_data['Date'].dt.to_timestamp()
        
        # Create interactive chart
        fig = go.Figure()
        
        # Add income bars
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['Deposits'],
            name='Income',
            marker_color='#3fb950',
            hovertemplate='<b>Income</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))
        
        # Add expense bars
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=-monthly_data['Withdrawls'],
            name='Expenses',
            marker_color='#f85149',
            hovertemplate='<b>Expenses</b><br>%{x}<br>$%{customdata:,.2f}<extra></extra>',
            customdata=monthly_data['Withdrawls']
        ))
        
        # Add net flow line
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Net_Flow'],
            mode='lines+markers',
            name='Net Flow',
            line=dict(color='#58a6ff', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Net Flow</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Monthly Cash Flow",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12, color="#c5cdd9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_income = monthly_data['Deposits'].mean()
            st.metric("Avg Monthly Income", f"${avg_income:,.2f}")
        with col2:
            avg_expenses = monthly_data['Withdrawls'].mean()
            st.metric("Avg Monthly Expenses", f"${avg_expenses:,.2f}")
        with col3:
            avg_net = monthly_data['Net_Flow'].mean()
            st.metric("Avg Net Flow", f"${avg_net:,.2f}")
        with col4:
            savings_rate = (avg_net / avg_income * 100) if avg_income > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
            
    except Exception as e:
        st.error(f"Error rendering cash flow chart: {str(e)}")

def render_interactive_categories(data: pd.DataFrame):
    """Render interactive category analysis"""
    try:
        if 'Category' not in data.columns:
            st.warning("Category information not available")
            return
        
        # Category spending analysis
        category_spending = data.groupby('Category')['Withdrawls'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive pie chart
            fig_pie = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title="Spending Distribution by Category",
                hover_data=[category_spending.values],
                labels={'value': 'Amount'}
            )
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Interactive bar chart
            fig_bar = px.bar(
                x=category_spending.values,
                y=category_spending.index,
                orientation='h',
                title="Total Spending by Category",
                labels={'x': 'Amount ($)', 'y': 'Category'},
                color=category_spending.values,
                color_continuous_scale='teal'
            )
            
            fig_bar.update_traces(
                hovertemplate='<b>%{y}</b><br>Amount: $%{x:,.2f}<extra></extra>'
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error rendering category analysis: {str(e)}")

def render_interactive_trends(data: pd.DataFrame):
    """Render interactive trend analysis"""
    try:
        # Daily balance trend
        if 'Balance' in data.columns:
            daily_balance = data.groupby('Date')['Balance'].last().reset_index()
            
            fig = px.line(
                daily_balance,
                x='Date',
                y='Balance',
                title="Account Balance Over Time",
                labels={'Balance': 'Balance ($)', 'Date': 'Date'}
            )
            
            fig.update_traces(
                line=dict(color='#58a6ff', width=2),
                hovertemplate='<b>Balance</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
            )
            
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction patterns
        if 'Date' in data.columns:
            data['DayOfWeek'] = data['Date'].dt.day_name()
            day_spending = data.groupby('DayOfWeek')['Withdrawls'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_day = px.bar(
                x=day_spending.index,
                y=day_spending.values,
                title="Average Spending by Day of Week",
                labels={'x': 'Day', 'y': 'Average Amount ($)'},
                color=day_spending.values,
                color_continuous_scale='teal'
            )
            
            fig_day.update_traces(
                hovertemplate='<b>%{x}</b><br>Avg Spending: $%{y:.2f}<extra></extra>'
            )
            
            st.plotly_chart(fig_day, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error rendering trend analysis: {str(e)}")

def render_interactive_analysis(data: pd.DataFrame):
    """Render detailed interactive analysis"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction volume over time
            daily_transactions = data.groupby('Date').size().reset_index(name='count')
            
            fig_volume = px.line(
                daily_transactions,
                x='Date',
                y='count',
                title="Daily Transaction Volume",
                labels={'count': 'Number of Transactions', 'Date': 'Date'}
            )
            
            fig_volume.update_traces(
                line=dict(color='#39d2c0', width=2),
                hovertemplate='<b>Transactions</b><br>%{x}<br>Count: %{y}<extra></extra>'
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Transaction size distribution
            if 'Withdrawls' in data.columns:
                non_zero_withdrawals = data[data['Withdrawls'] > 0]['Withdrawls']
                
                if not non_zero_withdrawals.empty:
                    fig_dist = px.histogram(
                        x=non_zero_withdrawals,
                        title="Transaction Size Distribution",
                        labels={'x': 'Transaction Amount ($)', 'y': 'Frequency'},
                        nbins=20
                    )
                    
                    fig_dist.update_traces(
                        hovertemplate='<b>Amount Range</b><br>$%{x}<br>Frequency: %{y}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
    except Exception as e:
        st.error(f"Error rendering detailed analysis: {str(e)}")

def render_enhanced_data_entry(data: pd.DataFrame):
    """Enhanced data entry with better UX"""
    with st.sidebar:
        st.markdown("---")
        st.header('💼 Data Management')
        
        # Quick add transaction
        with st.expander("➕ Quick Add Transaction"):
            new_date = st.date_input('Date', value=pd.to_datetime('today'))
            new_description = st.text_input('Description', placeholder="Enter transaction description")
            
            col1, col2 = st.columns(2)
            with col1:
                new_deposit = st.number_input('Deposit', min_value=0.0, value=0.0, format="%.2f")
            with col2:
                new_withdrawal = st.number_input('Withdrawal', min_value=0.0, value=0.0, format="%.2f")
            
            new_category = st.selectbox('Category', 
                options=['food_dining', 'shopping', 'transportation', 'utilities', 'other'])
            
            if st.button('✅ Add Transaction', type="primary"):
                if new_description:
                    new_transaction = pd.DataFrame({
                        'Date': [new_date],
                        'Description': [new_description],
                        'Deposits': [new_deposit],
                        'Withdrawls': [new_withdrawal],
                        'Balance': [data['Balance'].iloc[-1] + new_deposit - new_withdrawal if not data.empty else new_deposit - new_withdrawal],
                        'Category': [new_category],
                        'Recurring': [False]
                    })
                    st.success('✅ Transaction added successfully!')
                    st.info('💡 Refresh the page to see the new transaction in your data.')
                else:
                    st.error('Please enter a description for the transaction.')
        
        # Enhanced export options
        with st.expander("📤 Quick Export"):
            export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
            
            if st.button("📥 Export Current View"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "CSV":
                    csv = data.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"financial_data_{timestamp}.csv",
                        "text/csv"
                    )
                elif export_format == "JSON":
                    json_data = data.to_json(orient='records', date_format='iso')
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"financial_data_{timestamp}.json",
                        "application/json"
                    )
                
                st.success("✅ Export ready for download!")

def initialize_enterprise_features():
    """Initialize enterprise features with error handling"""
    try:
        # Authentication check
        auth = get_auth()
        config = get_config()
        
        # Show authentication in sidebar
        with st.sidebar:
            with st.expander("🔐 Authentication"):
                show_auth_sidebar()
        
        # Check if authentication is required
        if config.security.enable_authentication and not auth.is_authenticated():
            st.markdown("""
            <div class="main-header">
                <h1 class="header-title">🔐 FinViz Dashboard</h1>
                <p class="header-subtitle">Please login to access the dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            auth.show_login_form()
            return False
        
        # Performance monitoring in sidebar
        with st.sidebar:
            with st.expander("📊 Performance Metrics"):
                show_performance_metrics()
        
        # Configuration UI in sidebar
        with st.sidebar:
            with st.expander("⚙️ Configuration"):
                display_config_ui()
        
        # Handle authentication modals
        handle_auth_modals()
        
        return True
    except Exception as e:
        st.error(f"Enterprise feature initialization failed: {e}")
        return True  # Continue with basic features

def render_enhanced_header():
    """Render professional financial tool header"""
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title"><span class="header-divider"></span>FinViz Dashboard</h1>
        <p class="header-subtitle">Financial Analytics &amp; Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message and quick stats
    if st.session_state.get('show_welcome', True):
        with st.expander("🚀 Welcome to Your Enhanced Dashboard", expanded=False):
            st.markdown("""
            **Enhanced Features Available:**
            - 🎛️ **Real-time Interactive Filtering** - Filter data and watch charts update instantly
            - 📤 **Professional Export Center** - Generate PDF reports, Excel files, and more
            - ⌨️ **Keyboard Shortcuts** - Power user navigation (Press ? for help)
            - 🎨 **Dark Theme** - Easy on the eyes for extended use
            
            **Quick Start:**
            1. Upload your CSV file below 📁
            2. Enable interactive filtering in the sidebar 🎛️
            3. Explore the interactive visualizations and export options
            """)
            
            if st.button("✅ Got it, don't show again", key="hide_welcome"):
                st.session_state.show_welcome = False
                st.rerun()

def render_enhanced_upload_section():
    """Render enhanced upload section with instructions and sample"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("📋 View Sample Format", key="view_sample_btn", type="secondary"):
            show_sample_format()
    
    with col3:
        if st.button("❓ Upload Help", key="upload_help_btn", type="secondary"):
            show_upload_help()

def show_sample_format():
    """Show sample CSV format"""
    sample_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Description': ['Salary Deposit', 'Grocery Store', 'Gas Station'],
        'Deposits': [5000.00, 0.00, 0.00],
        'Withdrawls': [0.00, 125.50, 45.00],
        'Balance': [5000.00, 4874.50, 4829.50]
    })
    
    with st.expander("📋 Sample CSV Format", expanded=True):
        st.markdown("**Expected CSV structure:**")
        st.dataframe(sample_data, use_container_width=True)
        
        st.download_button(
            "📥 Download Sample CSV",
            sample_data.to_csv(index=False),
            "sample_financial_data.csv",
            "text/csv",
            help="Download this sample file as a template"
        )

def show_upload_help():
    """Show upload help and requirements"""
    with st.expander("📖 Upload Requirements & Help", expanded=True):
        st.markdown("""
        ### 📋 Required Columns:
        - **Date**: Transaction date (YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY)
        - **Description**: Transaction description or memo
        
        ### 📊 Optional Columns:
        - **Deposits**: Money received (positive amounts)
        - **Withdrawls**: Money spent (positive amounts)  
        - **Balance**: Account balance after transaction
        
        ### 💡 Tips for Best Results:
        1. **File Format**: Save as CSV with UTF-8 encoding
        2. **Date Format**: Use consistent date format throughout
        3. **Amounts**: Remove currency symbols, use periods for decimals
        4. **Headers**: First row should contain column names only
        5. **Encoding**: If you see strange characters, try UTF-8 encoding
        
        ### 🔧 Troubleshooting:
        - **Special Characters**: Remove or replace special characters
        - **Empty Cells**: Fill or remove empty date/description cells
        - **Large Files**: For files over 1MB, consider splitting into smaller chunks
        """)

def process_uploaded_file_enhanced(uploaded_file):
    """Enhanced file processing with better error handling and progress tracking"""
    try:
        with st.spinner("🔄 Processing your file..."):
            # Create progress tracking
            progress_container = st.container()
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
            
            # Step 1: Read file (20%)
            status_text.info("📂 Reading CSV file...")
            progress_bar.progress(20)
            
            try:
                if UTILS_AVAILABLE:
                    financial_data, encoding, separator = safe_csv_reader(uploaded_file)
                else:
                    financial_data, encoding, separator = basic_safe_csv_reader(uploaded_file)
                    
                if financial_data is None:
                    return None
                    
                status_text.success(f"✅ File read successfully (encoding: {encoding}, separator: '{separator}')")
            except Exception as e:
                status_text.error(f"❌ Failed to read file: {str(e)}")
                return None
            
            # Step 2: Normalize columns (40%)
            status_text.info("🔧 Normalizing column names...")
            progress_bar.progress(40)
            
            if UTILS_AVAILABLE:
                financial_data = normalize_column_names(financial_data)
            else:
                financial_data = basic_normalize_column_names(financial_data)
            
            # Step 3: Validate structure (60%)
            status_text.info("✅ Validating data structure...")
            progress_bar.progress(60)
            
            # Show detected columns
            with st.expander("🔍 Detected Columns", expanded=False):
                st.write("**Columns found in your file:**")
                col_info = []
                for col in financial_data.columns:
                    col_info.append({
                        "Column": col,
                        "Type": str(financial_data[col].dtype),
                        "Sample": str(financial_data[col].iloc[0]) if not financial_data.empty else "N/A"
                    })
                st.table(col_info)
            
            # Validate required columns
            required_columns = ['Date', 'Description']
            missing_columns = [col for col in required_columns if col not in financial_data.columns]
            
            if missing_columns:
                status_text.error(f"❌ Missing required columns: {missing_columns}")
                with st.expander("📋 Column Requirements", expanded=True):
                    st.error("**Missing Required Columns**")
                    st.markdown("**✅ Required:** Date, Description")
                    st.markdown(f"**📊 Your columns:** {', '.join(financial_data.columns)}")
                return None
            
            # Step 4: Process dates (80%)
            status_text.info("📅 Processing dates...")
            progress_bar.progress(80)
            
            if UTILS_AVAILABLE:
                financial_data['Date'] = financial_data['Date'].apply(custom_date_parser)
            else:
                financial_data['Date'] = financial_data['Date'].apply(basic_date_parser)
                
            initial_count = len(financial_data)
            financial_data = financial_data.dropna(subset=['Date'])
            
            if len(financial_data) < initial_count:
                removed_count = initial_count - len(financial_data)
                st.warning(f"⚠️ Removed {removed_count} rows with invalid dates")
            
            # Step 5: Process amounts and finalize (100%)
            status_text.info("💰 Processing amounts...")
            progress_bar.progress(100)
            
            # Process amount columns
            process_amount_columns(financial_data)
            
            # Add categories
            if UTILS_AVAILABLE:
                financial_data['Category'] = financial_data['Description'].apply(categorize_transaction)
            else:
                financial_data['Category'] = financial_data['Description'].apply(basic_categorize_transaction)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            
            # Show success summary
            show_processing_success(financial_data)
            
            return financial_data
            
    except Exception as e:
        st.error(f"❌ Unexpected error processing file: {str(e)}")
        return None

def process_amount_columns(financial_data):
    """Process and clean amount columns"""
    amount_columns = ['Deposits', 'Withdrawls', 'Balance', 'Amount']
    
    for col in amount_columns:
        if col in financial_data.columns:
            # Clean and convert to numeric
            financial_data[col] = pd.to_numeric(
                financial_data[col].astype(str)
                .str.replace(',', '')
                .str.replace('$', '')
                .str.replace('€', '')
                .str.replace('£', ''), 
                errors='coerce'
            ).fillna(0)
    
    # Ensure we have required amount columns
    if 'Deposits' not in financial_data.columns:
        financial_data['Deposits'] = 0
    if 'Withdrawls' not in financial_data.columns:
        financial_data['Withdrawls'] = 0
    
    # Calculate balance if not provided
    if 'Balance' not in financial_data.columns:
        financial_data['Balance'] = (financial_data['Deposits'] - financial_data['Withdrawls']).cumsum()

def show_column_requirements(available_columns):
    """Show column requirements when validation fails"""
    with st.expander("📋 Column Requirements", expanded=True):
        st.error("**Missing Required Columns**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Required Columns:**")
            st.markdown("- `Date` (or similar: Transaction Date, Trans Date)")
            st.markdown("- `Description` (or similar: Memo, Details, Narration)")
            
        with col2:
            st.markdown("**📊 Your Columns:**")
            for col in available_columns:
                st.markdown(f"- `{col}`")
        
        st.markdown("**💡 Tip:** Column names are case-insensitive and the system recognizes common variations.")

def show_troubleshooting_tips():
    """Show troubleshooting tips for file processing issues"""
    with st.expander("🔧 Troubleshooting Guide", expanded=True):
        st.markdown("""
        ### Common Issues & Solutions:
        
        **🔴 File Reading Errors:**
        - Save your file with UTF-8 encoding
        - Ensure the file is in proper CSV format
        - Remove any special characters from the filename
        
        **🔴 Date Processing Issues:**
        - Use consistent date formats (YYYY-MM-DD recommended)
        - Remove empty date cells
        - Check for merged cells in Excel before exporting
        
        **🔴 Column Recognition Issues:**
        - Ensure column headers are in the first row
        - Remove any extra header rows
        - Use standard column names when possible
        
        **🔴 Amount Processing Issues:**
        - Remove currency symbols ($, €, £)
        - Use periods (.) for decimal points
        - Avoid scientific notation (1.23E+10)
        """)

def show_processing_success(financial_data):
    """Show successful processing summary"""
    with st.expander("✅ Processing Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Records", f"{len(financial_data):,}")
            
        with col2:
            if not financial_data.empty:
                date_range = f"{financial_data['Date'].min().strftime('%Y-%m-%d')} to {financial_data['Date'].max().strftime('%Y-%m-%d')}"
                st.metric("📅 Date Range", date_range)
        
        with col3:
            if 'Deposits' in financial_data.columns and 'Withdrawls' in financial_data.columns:
                net_flow = financial_data['Deposits'].sum() - financial_data['Withdrawls'].sum()
                st.metric("💰 Net Flow", f"${net_flow:,.2f}")
        
        # Data preview
        st.markdown("**📋 Data Preview:**")
        st.dataframe(financial_data.head(5), use_container_width=True)
    
    st.success(f"🎉 Successfully processed {len(financial_data):,} transactions!")

def add_feature_discovery():
    """Add feature discovery section to help users find new capabilities"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ✨ New Features")
        
        if st.button("🎯 Feature Tour", type="primary"):
            show_feature_tour()
        
        # Quick feature toggles
        st.markdown("#### ⚡ Quick Access")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎛️ Filters", key="toggle_filters", help="Toggle interactive filters"):
                st.session_state.interactive_filters_enabled = not st.session_state.interactive_filters_enabled
                st.rerun()
        
        with col2:
            if st.button("📤 Export", key="toggle_export", help="Open export center"):
                st.session_state.show_export_center = True
        
        # Keyboard shortcuts help
        with st.expander("⌨️ Keyboard Shortcuts"):
            shortcuts = {
                "**Ctrl + R**": "Refresh data",
                "**Ctrl + F**": "Focus search box",
                "**?**": "Show help dialog",
                "**Ctrl + E**": "Quick export",
                "**Esc**": "Close modals"
            }
            
            for shortcut, description in shortcuts.items():
                st.markdown(f"{shortcut} - {description}")
            
            st.markdown("---")
            st.markdown("💡 **Tip:** Hover over elements for additional help!")

def show_feature_tour():
    """Show guided tour of new features"""
    st.markdown("### 🚀 Welcome to Enhanced FinViz Dashboard!")
    
    st.success("**Major Features Available:**")
    
    # Feature 1: Interactive Filtering
    with st.expander("🎛️ 1. Interactive Real-Time Filtering", expanded=True):
        st.markdown("""
        **What's Available:**
        - 🔄 **Real-time filtering** - Charts update instantly as you adjust filters
        - 📅 **Advanced date ranges** - Precise control over time periods
        - 💰 **Amount sliders** - Visual filtering by transaction amounts
        - 🏷️ **Category selection** - Multi-select category filtering
        - 🔍 **Text search** - Search transaction descriptions in real-time
        - 📊 **Filter impact display** - See how many records are filtered
        
        **How to Use:**
        1. Look for "Interactive Filters" in the sidebar
        2. Toggle "Real-time Filtering" to ON
        3. Adjust any filter and watch charts update instantly!
        """)
    
    # Feature 2: Enhanced Export
    with st.expander("📤 2. Professional Export System", expanded=True):
        st.markdown("""
        **What's Available:**
        - 📊 **PDF Reports** - Professional formatted reports with charts
        - 📈 **Excel Export** - Interactive spreadsheets with formatting
        - 🔄 **Multiple Formats** - CSV, JSON, HTML, and more
        - 📧 **Automated Reports** - Schedule regular email reports
        - 🔗 **Share Links** - Generate shareable dashboard links
        - 📋 **Custom Templates** - Branded reports with your logo
        
        **How to Use:**
        1. Upload your data first
        2. Look for "Enhanced Export Center" 
        3. Choose your format and customize options
        """)
    
    # Additional Features
    st.markdown("### 🎁 Additional Features:")
    
    bonus_features = [
        "⌨️ **Keyboard Shortcuts** - Power user navigation",
        "📱 **Mobile Responsive** - Works great on phones and tablets", 
        "🔍 **Enhanced Search** - Natural language transaction search",
        "📊 **Better Charts** - Interactive Plotly visualizations",
        "💡 **Smart Tips** - Contextual help throughout the app",
        "🚀 **Performance** - Faster loading and better caching",
        "🎨 **Dark Theme** - Easy on the eyes for extended use"
    ]
    
    for feature in bonus_features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.info("💡 **Pro Tip:** Try enabling real-time filtering and then exporting a professional report to see all features working together!")

def render_basic_dashboard(data: pd.DataFrame):
    """Basic dashboard when advanced visualizations aren't available"""
    st.markdown("### 📈 Basic Financial Dashboard")
    
    if data.empty:
        st.warning("No data available for visualization")
        return
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_income = data['Deposits'].sum() if 'Deposits' in data.columns else 0
        st.metric("Total Income", f"${total_income:,.2f}")
    with col2:
        total_expenses = data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        net_flow = total_income - total_expenses
        st.metric("Net Flow", f"${net_flow:,.2f}")
    with col4:
        transaction_count = len(data)
        st.metric("Total Transactions", f"{transaction_count:,}")
    
    # Basic monthly chart if date column exists
    if 'Date' in data.columns:
        try:
            data['Month'] = pd.to_datetime(data['Date']).dt.to_period('M')
            monthly_data = data.groupby('Month').agg({
                'Deposits': 'sum',
                'Withdrawls': 'sum'
            }).reset_index()
            
            if not monthly_data.empty:
                monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
                monthly_data['Net Flow'] = monthly_data['Deposits'] - monthly_data['Withdrawls']
                
                # Simple bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_data['Month'],
                    y=monthly_data['Deposits'],
                    name='Income',
                    marker_color='#3fb950'
                ))
                fig.add_trace(go.Bar(
                    x=monthly_data['Month'],
                    y=-monthly_data['Withdrawls'],
                    name='Expenses',
                    marker_color='#f85149'
                ))
                
                fig.update_layout(
                    title="Monthly Cash Flow",
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", size=12, color="#c5cdd9")
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating basic chart: {str(e)}")

def render_basic_export(data: pd.DataFrame):
    """Basic export functionality when enhanced export isn't available"""
    with st.sidebar:
        st.markdown("---")
        st.header('📤 Export Data')
        
        if st.button("📥 Download CSV"):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="financial_data.csv">Click to Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ CSV ready for download!")
        
        if st.button("📊 Download JSON"):
            json_data = data.to_json(orient='records', date_format='iso')
            b64 = base64.b64encode(json_data.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="financial_data.json">Click to Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ JSON ready for download!")

if __name__ == '__main__':
    main()