import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class UserInputHandler:
    """Handle user inputs and preferences for the finance dashboard"""
    
    def __init__(self):
        self.user_preferences_file = "user_preferences.json"
        self.load_user_preferences()
    
    def load_user_preferences(self):
        """Load user preferences from file"""
        try:
            if os.path.exists(self.user_preferences_file):
                with open(self.user_preferences_file, 'r') as f:
                    self.preferences = json.load(f)
            else:
                self.preferences = self.get_default_preferences()
        except Exception as e:
            st.warning(f"Could not load user preferences: {e}")
            self.preferences = self.get_default_preferences()
    
    def save_user_preferences(self):
        """Save user preferences to file"""
        try:
            with open(self.user_preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Could not save user preferences: {e}")
    
    def get_default_preferences(self):
        """Get default user preferences"""
        return {
            "budget_categories": {
                "Food & Dining": 500.0,
                "Shopping": 300.0,
                "Transportation": 200.0,
                "Entertainment": 150.0,
                "Utilities": 400.0,
                "Healthcare": 200.0,
                "Other": 100.0
            },
            "favorite_widgets": [
                "Cash Flow",
                "Budget Tracking",
                "🧠 NLP Analysis"
            ],
            "default_chart_type": "Line Chart",
            "currency_symbol": "$",
            "date_range_days": 30,
            "auto_categorize": True,
            "show_anomalies": True,
            "nlp_insights": True
        }
    
    def display_user_settings(self):
        """Display user settings and preferences in sidebar"""
        st.sidebar.header("⚙️ User Settings")
        
        # Budget preferences
        with st.sidebar.expander("💰 Budget Settings", expanded=False):
            st.write("Set your monthly budget limits:")
            budget_categories = self.preferences.get("budget_categories", {})
            
            for category in budget_categories:
                new_budget = st.number_input(
                    f"{category}",
                    min_value=0.0,
                    value=float(budget_categories[category]),
                    key=f"budget_{category}"
                )
                budget_categories[category] = new_budget
            
            # Add new category
            new_category = st.text_input("Add new budget category:")
            new_budget_amount = st.number_input("Budget amount:", min_value=0.0, value=0.0)
            
            if st.button("Add Category") and new_category:
                budget_categories[new_category] = new_budget_amount
                st.success(f"Added {new_category} with budget ${new_budget_amount}")
        
        # Display preferences
        with st.sidebar.expander("🎨 Display Preferences", expanded=False):
            self.preferences["default_chart_type"] = st.selectbox(
                "Default Chart Type",
                ["Line Chart", "Bar Chart", "Histogram"],
                index=["Line Chart", "Bar Chart", "Histogram"].index(
                    self.preferences.get("default_chart_type", "Line Chart")
                )
            )
            
            self.preferences["currency_symbol"] = st.text_input(
                "Currency Symbol",
                value=self.preferences.get("currency_symbol", "$")
            )
            
            self.preferences["date_range_days"] = st.slider(
                "Default Date Range (days)",
                min_value=7,
                max_value=365,
                value=self.preferences.get("date_range_days", 30)
            )
        
        # Feature preferences
        with st.sidebar.expander("🔧 Feature Settings", expanded=False):
            self.preferences["auto_categorize"] = st.checkbox(
                "Auto-categorize transactions",
                value=self.preferences.get("auto_categorize", True)
            )
            
            self.preferences["show_anomalies"] = st.checkbox(
                "Show anomaly detection",
                value=self.preferences.get("show_anomalies", True)
            )
            
            self.preferences["nlp_insights"] = st.checkbox(
                "Enable NLP insights",
                value=self.preferences.get("nlp_insights", True)
            )
        
        # Save preferences
        if st.sidebar.button("💾 Save Preferences"):
            self.save_user_preferences()
            st.sidebar.success("Preferences saved!")
        
        # Reset to defaults
        if st.sidebar.button("🔄 Reset to Defaults"):
            self.preferences = self.get_default_preferences()
            st.sidebar.success("Preferences reset to defaults!")
    
    def get_user_filters(self, data):
        """Get user-defined filters for data"""
        st.sidebar.header("🔍 Data Filters")
        
        filters = {}
        
        # Date range filter
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            min_date = data['Date'].min().date()
            max_date = data['Date'].max().date()
            
            default_start = max_date - timedelta(days=self.preferences.get("date_range_days", 30))
            default_start = max(default_start, min_date)
            
            filters['start_date'] = st.sidebar.date_input(
                "Start Date",
                value=default_start,
                min_value=min_date,
                max_value=max_date
            )
            
            filters['end_date'] = st.sidebar.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Category filter
        if 'Category' in data.columns:
            categories = data['Category'].unique().tolist()
            filters['categories'] = st.sidebar.multiselect(
                "Select Categories",
                options=categories,
                default=categories
            )
        
        # Amount filters with error handling
        if 'Withdrawls' in data.columns:
            min_withdrawal = float(data['Withdrawls'].min())
            max_withdrawal = float(data['Withdrawls'].max())
            
            # Only show slider if there's a meaningful range
            if max_withdrawal > min_withdrawal:
                filters['withdrawal_range'] = st.sidebar.slider(
                    "Withdrawal Amount Range",
                    min_value=min_withdrawal,
                    max_value=max_withdrawal,
                    value=(min_withdrawal, max_withdrawal),
                    format=f"{self.preferences.get('currency_symbol', '$')}%.2f"
                )
            else:
                # If all values are the same, just display the value and set default range
                if max_withdrawal > 0:
                    st.sidebar.write(f"📊 All withdrawals: {self.preferences.get('currency_symbol', '$')}{max_withdrawal:.2f}")
                else:
                    st.sidebar.write("📊 No withdrawal transactions found")
                filters['withdrawal_range'] = (min_withdrawal, max_withdrawal)
        
        if 'Deposits' in data.columns:
            min_deposit = float(data['Deposits'].min())
            max_deposit = float(data['Deposits'].max())
            
            # Only show slider if there's a meaningful range
            if max_deposit > min_deposit:
                filters['deposit_range'] = st.sidebar.slider(
                    "Deposit Amount Range",
                    min_value=min_deposit,
                    max_value=max_deposit,
                    value=(min_deposit, max_deposit),
                    format=f"{self.preferences.get('currency_symbol', '$')}%.2f"
                )
            else:
                # If all values are the same, just display the value and set default range
                if max_deposit > 0:
                    st.sidebar.write(f"📊 All deposits: {self.preferences.get('currency_symbol', '$')}{max_deposit:.2f}")
                else:
                    st.sidebar.write("📊 No deposit transactions found")
                filters['deposit_range'] = (min_deposit, max_deposit)
        
        # Search filter
        filters['search_query'] = st.sidebar.text_input(
            "Search in descriptions",
            placeholder="Enter keywords or phrases..."
        )
        
        # Advanced search options
        with st.sidebar.expander("🔍 Advanced Search", expanded=False):
            filters['search_type'] = st.selectbox(
                "Search Type",
                ["Keywords", "🧠 Smart Search", "Regex Pattern"]
            )
            
            if filters['search_type'] == "🧠 Smart Search":
                st.info("💡 Smart search uses NLP to understand your query better!")
            elif filters['search_type'] == "Regex Pattern":
                st.info("🔍 Use regex patterns for advanced text matching")
        
        return filters
    
    def apply_filters(self, data, filters):
        """Apply user-defined filters to the data"""
        filtered_data = data.copy()
        
        # Apply date filters
        if 'start_date' in filters and 'end_date' in filters and 'Date' in data.columns:
            filtered_data = filtered_data[
                (filtered_data['Date'] >= pd.to_datetime(filters['start_date'])) &
                (filtered_data['Date'] <= pd.to_datetime(filters['end_date']))
            ]
        
        # Apply category filter
        if 'categories' in filters and filters['categories'] and 'Category' in data.columns:
            filtered_data = filtered_data[filtered_data['Category'].isin(filters['categories'])]
        
        # Apply amount filters
        if 'withdrawal_range' in filters and 'Withdrawls' in data.columns:
            min_w, max_w = filters['withdrawal_range']
            filtered_data = filtered_data[
                (filtered_data['Withdrawls'] >= min_w) &
                (filtered_data['Withdrawls'] <= max_w)
            ]
        
        if 'deposit_range' in filters and 'Deposits' in data.columns:
            min_d, max_d = filters['deposit_range']
            filtered_data = filtered_data[
                (filtered_data['Deposits'] >= min_d) &
                (filtered_data['Deposits'] <= max_d)
            ]
        
        # Apply search filter
        if 'search_query' in filters and filters['search_query'] and 'Description' in data.columns:
            query = filters['search_query']
            search_type = filters.get('search_type', 'Keywords')
            
            if search_type == "Keywords":
                filtered_data = filtered_data[
                    filtered_data['Description'].str.contains(query, case=False, na=False)
                ]
            elif search_type == "🧠 Smart Search":
                # Use NLP-powered search if available
                from utils import smart_search_transactions
                filtered_data = smart_search_transactions(filtered_data, query)
            elif search_type == "Regex Pattern":
                try:
                    filtered_data = filtered_data[
                        filtered_data['Description'].str.contains(query, case=False, na=False, regex=True)
                    ]
                except Exception as e:
                    st.sidebar.error(f"Invalid regex pattern: {e}")
        
        return filtered_data
    
    def get_widget_selection(self):
        """Get user's widget selection for dashboard"""
        st.sidebar.header("📊 Dashboard Widgets")
        
        available_widgets = {
            "Cash Flow": "📈 Monthly cash flow analysis",
            "Budget Tracking": "💰 Budget vs actual spending",
            "Spending Patterns and Predictions": "🔮 ML-powered spending predictions",
            "🧠 NLP Analysis": "🤖 AI-powered transaction insights"
        }
        
        # Show widget descriptions
        with st.sidebar.expander("Widget Information", expanded=False):
            for widget, description in available_widgets.items():
                st.write(f"**{widget}**: {description}")
        
        # Widget selection
        favorite_widgets = self.preferences.get("favorite_widgets", list(available_widgets.keys()))
        
        selected_widgets = st.sidebar.multiselect(
            "Select Widgets to Display",
            options=list(available_widgets.keys()),
            default=[w for w in favorite_widgets if w in available_widgets.keys()]
        )
        
        # Save as favorites
        if st.sidebar.button("⭐ Save as Favorites"):
            self.preferences["favorite_widgets"] = selected_widgets
            st.sidebar.success("Favorites updated!")
        
        return selected_widgets
    
    def get_export_options(self):
        """Get user's export preferences"""
        st.sidebar.header("📤 Export Options")
        
        export_options = {}
        
        with st.sidebar.expander("Export Settings", expanded=False):
            export_options['format'] = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "JSON"]
            )
            
            export_options['include_metadata'] = st.checkbox(
                "Include metadata",
                value=True
            )
            
            export_options['date_format'] = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"]
            )
            
            export_options['filename_prefix'] = st.text_input(
                "Filename Prefix",
                value="financial_data"
            )
        
        return export_options
    
    def display_quick_actions(self):
        """Display quick action buttons"""
        st.sidebar.header("⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col2:
            if st.button("📊 View Summary"):
                return "show_summary"
        
        if st.sidebar.button("🧠 Generate AI Insights"):
            return "generate_insights"
        
        if st.sidebar.button("📈 Trend Analysis"):
            return "trend_analysis"
        
        return None

# Global instance
user_input_handler = UserInputHandler()