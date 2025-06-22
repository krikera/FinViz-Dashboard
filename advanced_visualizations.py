"""
Advanced Visualization Components for FinViz Dashboard
Enterprise-grade interactive charts, real-time dashboards, and advanced analytics visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.figure_factory as ff
    PLOTLY_FACTORY_AVAILABLE = True
except ImportError:
    PLOTLY_FACTORY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

class AdvancedVisualizationEngine:
    """Advanced visualization engine with interactive and real-time capabilities"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chart_themes = {
            'default': 'plotly',
            'dark': 'plotly_dark',
            'minimal': 'plotly_white',
            'presentation': 'presentation'
        }
    
    def create_executive_dashboard(self, financial_data: pd.DataFrame) -> None:
        """Create comprehensive executive dashboard with KPIs and trends"""
        st.title("📊 Executive Financial Dashboard")
        
        # Calculate KPIs
        kpis = self._calculate_kpis(financial_data)
        
        # Display KPI metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Balance",
                value=f"${kpis['current_balance']:,.2f}",
                delta=f"${kpis['balance_change']:,.2f}"
            )
        
        with col2:
            st.metric(
                label="📈 Monthly Income",
                value=f"${kpis['monthly_income']:,.2f}",
                delta=f"{kpis['income_change']:+.1f}%"
            )
        
        with col3:
            st.metric(
                label="📉 Monthly Expenses",
                value=f"${kpis['monthly_expenses']:,.2f}",
                delta=f"{kpis['expense_change']:+.1f}%"
            )
        
        with col4:
            st.metric(
                label="💡 Savings Rate",
                value=f"{kpis['savings_rate']:.1f}%",
                delta=f"{kpis['savings_change']:+.1f}%"
            )
        
        # Create interactive charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.create_cash_flow_waterfall(financial_data)
        
        with col2:
            self.create_category_sunburst(financial_data)
        
        # Time series analysis
        self.create_financial_trends_analysis(financial_data)
        
        # Advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            self.create_spending_heatmap(financial_data)
        
        with col2:
            self.create_transaction_distribution(financial_data)
    
    def create_cash_flow_waterfall(self, data: pd.DataFrame) -> None:
        """Create interactive waterfall chart for cash flow analysis"""
        st.subheader("💧 Cash Flow Waterfall")
        
        # Prepare data for waterfall chart
        monthly_data = self._prepare_monthly_data(data)
        
        if monthly_data.empty:
            st.warning("No data available for cash flow analysis")
            return
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Cash Flow", 
            orientation="v",
            measure=["relative"] * (len(monthly_data) - 1) + ["total"],
            x=monthly_data.index.strftime('%Y-%m'),
            textposition="outside",
            text=[f"${x:,.0f}" for x in monthly_data['net_flow']],
            y=monthly_data['net_flow'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.color_palette['success']}},
            decreasing={"marker": {"color": self.color_palette['danger']}},
            totals={"marker": {"color": self.color_palette['primary']}}
        ))
        
        fig.update_layout(
            title="Monthly Cash Flow Analysis",
            showlegend=False,
            height=400,
            xaxis_title="Month",
            yaxis_title="Amount ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_category_sunburst(self, data: pd.DataFrame) -> None:
        """Create interactive sunburst chart for expense categories"""
        st.subheader("☀️ Expense Breakdown")
        
        if 'Category' not in data.columns or data.empty:
            st.warning("No category data available")
            return
        
        # Prepare data for sunburst
        expense_data = data[data['Withdrawls'] > 0].copy()
        
        if expense_data.empty:
            st.warning("No expense data available")
            return
        
        category_summary = expense_data.groupby('Category')['Withdrawls'].sum().sort_values(ascending=False)
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=list(category_summary.index),
            parents=[""] * len(category_summary),
            values=list(category_summary.values),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=2,
        ))
        
        fig.update_layout(
            title="Expense Categories Distribution",
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_financial_trends_analysis(self, data: pd.DataFrame) -> None:
        """Create comprehensive financial trends analysis with multiple metrics"""
        st.subheader("📈 Financial Trends Analysis")
        
        # Create tabs for different trend analyses
        trend_tab1, trend_tab2, trend_tab3 = st.tabs(["📊 Overview", "🎯 Patterns", "🔮 Forecast"])
        
        with trend_tab1:
            self._create_overview_trends(data)
        
        with trend_tab2:
            self._create_pattern_analysis(data)
        
        with trend_tab3:
            self._create_forecast_analysis(data)
    
    def _create_overview_trends(self, data: pd.DataFrame) -> None:
        """Create overview trends visualization"""
        monthly_data = self._prepare_monthly_data(data)
        
        if monthly_data.empty:
            st.warning("No data available for trends analysis")
            return
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Monthly Cash Flow", "Income vs Expenses", "Balance Trend", "Transaction Volume"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12
        )
        
        # Monthly cash flow
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index.astype(str),
                y=monthly_data['net_flow'],
                mode='lines+markers',
                name='Net Flow',
                line=dict(color=self.color_palette['primary'], width=3),
                fill='tonexty' if monthly_data['net_flow'].iloc[0] >= 0 else 'tozeroy'
            ),
            row=1, col=1
        )
        
        # Income vs Expenses
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index.astype(str),
                y=monthly_data['total_income'],
                mode='lines+markers',
                name='Income',
                line=dict(color=self.color_palette['success'], width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index.astype(str),
                y=monthly_data['total_expenses'],
                mode='lines+markers',
                name='Expenses',
                line=dict(color=self.color_palette['danger'], width=2)
            ),
            row=1, col=2
        )
        
        # Balance trend
        if 'Balance' in data.columns:
            balance_trend = data.groupby(data['Date'].dt.to_period('M'))['Balance'].last()
            fig.add_trace(
                go.Scatter(
                    x=balance_trend.index.astype(str),
                    y=balance_trend.values,
                    mode='lines+markers',
                    name='Balance',
                    line=dict(color=self.color_palette['info'], width=3)
                ),
                row=2, col=1
            )
        
        # Transaction volume
        transaction_volume = data.groupby(data['Date'].dt.to_period('M')).size()
        fig.add_trace(
            go.Bar(
                x=transaction_volume.index.astype(str),
                y=transaction_volume.values,
                name='Transactions',
                marker_color=self.color_palette['secondary']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Financial Trends Overview",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_pattern_analysis(self, data: pd.DataFrame) -> None:
        """Create pattern analysis visualization"""
        col1, col2 = st.columns(2)
        
        with col1:
            self.create_spending_by_day_of_week(data)
        
        with col2:
            self.create_seasonal_analysis(data)
    
    def _create_forecast_analysis(self, data: pd.DataFrame) -> None:
        """Create forecast analysis with predictive modeling"""
        st.subheader("🔮 Financial Forecasting")
        
        # Simple trend-based forecasting
        monthly_data = self._prepare_monthly_data(data)
        
        if len(monthly_data) < 3:
            st.warning("Need at least 3 months of data for forecasting")
            return
        
        # Generate forecast
        forecast_periods = st.slider("Forecast Periods (months)", 1, 12, 6)
        forecast_data = self._generate_simple_forecast(monthly_data, forecast_periods)
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index.astype(str),
                y=monthly_data['net_flow'],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_palette['primary'], width=3)
            )
        )
        
        # Forecast data
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.astype(str),
                y=forecast_data['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_palette['warning'], width=2, dash='dash')
            )
        )
        
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.astype(str),
                y=forecast_data['upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.astype(str),
                y=forecast_data['lower'],
                mode='lines',
                line=dict(width=0),
                name='Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.2)',
                hoverinfo='skip'
            )
        )
        
        fig.update_layout(
            title="Cash Flow Forecast",
            xaxis_title="Period",
            yaxis_title="Net Cash Flow ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast insights
        st.subheader("📊 Forecast Insights")
        
        avg_forecast = forecast_data['forecast'].mean()
        current_avg = monthly_data['net_flow'].tail(3).mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Avg Monthly Flow",
                f"${avg_forecast:,.2f}",
                f"${avg_forecast - current_avg:+,.2f}"
            )
        
        with col2:
            st.metric(
                "6-Month Outlook",
                "Positive" if avg_forecast > 0 else "Negative",
                f"{((avg_forecast / abs(current_avg)) - 1) * 100:+.1f}%" if current_avg != 0 else "N/A"
            )
        
        with col3:
            trend = "Improving" if forecast_data['forecast'].iloc[-1] > forecast_data['forecast'].iloc[0] else "Declining"
            st.metric("Trend Direction", trend)
    
    def create_spending_heatmap(self, data: pd.DataFrame) -> None:
        """Create spending pattern heatmap"""
        st.subheader("🔥 Spending Heatmap")
        
        if data.empty:
            st.warning("No data available for heatmap")
            return
        
        # Prepare data for heatmap
        expense_data = data[data['Withdrawls'] > 0].copy()
        expense_data['DayOfWeek'] = expense_data['Date'].dt.day_name()
        expense_data['Hour'] = expense_data['Date'].dt.hour
        
        # Create pivot table for heatmap
        heatmap_data = expense_data.groupby(['DayOfWeek', 'Hour'])['Withdrawls'].sum().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Amount: $%{z:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Spending Patterns by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_transaction_distribution(self, data: pd.DataFrame) -> None:
        """Create transaction amount distribution analysis"""
        st.subheader("📊 Transaction Distribution")
        
        if data.empty:
            st.warning("No data available for distribution analysis")
            return
        
        # Create distribution plot
        expense_amounts = data[data['Withdrawls'] > 0]['Withdrawls']
        
        if expense_amounts.empty:
            st.warning("No expense data available")
            return
        
        # Create histogram with distribution curve
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Transaction Amount Distribution", "Box Plot Analysis"),
            vertical_spacing=0.15
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=expense_amounts,
                nbinsx=50,
                name='Distribution',
                marker_color=self.color_palette['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=expense_amounts,
                name='Box Plot',
                marker_color=self.color_palette['secondary'],
                boxpoints='outliers'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Transaction Amount ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"${expense_amounts.mean():.2f}")
        
        with col2:
            st.metric("Median", f"${expense_amounts.median():.2f}")
        
        with col3:
            st.metric("Std Dev", f"${expense_amounts.std():.2f}")
        
        with col4:
            st.metric("90th Percentile", f"${expense_amounts.quantile(0.9):.2f}")
    
    def create_spending_by_day_of_week(self, data: pd.DataFrame) -> None:
        """Create spending analysis by day of week"""
        st.subheader("📅 Weekly Spending Pattern")
        
        expense_data = data[data['Withdrawls'] > 0].copy()
        
        if expense_data.empty:
            st.warning("No expense data available")
            return
        
        expense_data['DayOfWeek'] = expense_data['Date'].dt.day_name()
        daily_spending = expense_data.groupby('DayOfWeek')['Withdrawls'].agg(['sum', 'mean', 'count'])
        
        # Reorder by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_spending = daily_spending.reindex(day_order)
        
        # Create polar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=daily_spending['sum'],
            theta=daily_spending.index,
            fill='toself',
            name='Total Spending',
            line_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, daily_spending['sum'].max() * 1.1]
                )),
            title="Weekly Spending Pattern",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_seasonal_analysis(self, data: pd.DataFrame) -> None:
        """Create seasonal spending analysis"""
        st.subheader("🌱 Seasonal Analysis")
        
        if data.empty:
            st.warning("No data available for seasonal analysis")
            return
        
        # Prepare seasonal data
        data_copy = data.copy()
        data_copy['Month'] = data_copy['Date'].dt.month_name()
        data_copy['Quarter'] = data_copy['Date'].dt.quarter
        
        monthly_expenses = data_copy[data_copy['Withdrawls'] > 0].groupby('Month')['Withdrawls'].sum()
        
        # Reorder months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_expenses = monthly_expenses.reindex(month_order).fillna(0)
        
        # Create seasonal chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_expenses.index,
            y=monthly_expenses.values,
            marker_color=self.color_palette['info'],
            name='Monthly Expenses'
        ))
        
        fig.update_layout(
            title="Seasonal Spending Patterns",
            xaxis_title="Month",
            yaxis_title="Total Expenses ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_interactive_portfolio_view(self, data: pd.DataFrame) -> None:
        """Create interactive portfolio-style view of financial data"""
        st.title("📈 Portfolio View")
        
        # Create portfolio-style metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._create_performance_gauge(data)
        
        with col2:
            self._create_allocation_donut(data)
        
        with col3:
            self._create_risk_metrics(data)
        
        # Interactive filters
        st.subheader("🎛️ Interactive Analysis")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            date_range = st.date_input(
                "Select Date Range",
                value=(data['Date'].min().date(), data['Date'].max().date()),
                min_value=data['Date'].min().date(),
                max_value=data['Date'].max().date()
            )
        
        with filter_col2:
            if 'Category' in data.columns:
                unique_categories = data['Category'].unique().tolist()
                categories = st.multiselect(
                    "Select Categories",
                    options=unique_categories,
                    default=unique_categories
                )
            else:
                categories = []
        
        # Apply filters and update visualizations
        filtered_data = self._apply_portfolio_filters(data, date_range, categories)
        
        if not filtered_data.empty:
            self.create_financial_trends_analysis(filtered_data)
    
    def _calculate_kpis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key performance indicators"""
        if data.empty:
            return {
                'current_balance': 0,
                'balance_change': 0,
                'monthly_income': 0,
                'income_change': 0,
                'monthly_expenses': 0,
                'expense_change': 0,
                'savings_rate': 0,
                'savings_change': 0
            }
        
        # Current balance
        current_balance = data['Balance'].iloc[-1] if 'Balance' in data.columns else 0
        
        # Balance change (last 30 days)
        thirty_days_ago = data['Date'].max() - timedelta(days=30)
        recent_data = data[data['Date'] > thirty_days_ago]
        balance_change = recent_data['Balance'].iloc[-1] - recent_data['Balance'].iloc[0] if len(recent_data) > 0 and 'Balance' in data.columns else 0
        
        # Monthly income and expenses
        current_month = data['Date'].max().replace(day=1)
        monthly_data = data[data['Date'] >= current_month]
        
        monthly_income = monthly_data['Deposits'].sum() if 'Deposits' in data.columns else 0
        monthly_expenses = monthly_data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0
        
        # Previous month for comparison
        prev_month = (current_month - timedelta(days=1)).replace(day=1)
        prev_month_data = data[(data['Date'] >= prev_month) & (data['Date'] < current_month)]
        
        prev_monthly_income = prev_month_data['Deposits'].sum() if 'Deposits' in data.columns else 0
        prev_monthly_expenses = prev_month_data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0
        
        # Calculate changes
        income_change = ((monthly_income - prev_monthly_income) / prev_monthly_income * 100) if prev_monthly_income > 0 else 0
        expense_change = ((monthly_expenses - prev_monthly_expenses) / prev_monthly_expenses * 100) if prev_monthly_expenses > 0 else 0
        
        # Savings rate
        savings_rate = ((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0
        prev_savings_rate = ((prev_monthly_income - prev_monthly_expenses) / prev_monthly_income * 100) if prev_monthly_income > 0 else 0
        savings_change = savings_rate - prev_savings_rate
        
        return {
            'current_balance': current_balance,
            'balance_change': balance_change,
            'monthly_income': monthly_income,
            'income_change': income_change,
            'monthly_expenses': monthly_expenses,
            'expense_change': expense_change,
            'savings_rate': savings_rate,
            'savings_change': savings_change
        }
    
    def _prepare_monthly_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare monthly aggregated data for analysis"""
        if data.empty:
            return pd.DataFrame()
        
        # Group by month
        monthly_groups = data.groupby(data['Date'].dt.to_period('M'))
        
        monthly_data = pd.DataFrame({
            'total_income': monthly_groups['Deposits'].sum() if 'Deposits' in data.columns else 0,
            'total_expenses': monthly_groups['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0,
        })
        
        monthly_data['net_flow'] = monthly_data['total_income'] - monthly_data['total_expenses']
        
        return monthly_data
    
    def _generate_simple_forecast(self, historical_data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Generate simple trend-based forecast"""
        if len(historical_data) < 2:
            return pd.DataFrame()
        
        # Simple linear trend
        x = np.arange(len(historical_data))
        y = historical_data['net_flow'].values
        
        # Linear regression
        z = np.polyfit(x, y, 1)
        trend = np.poly1d(z)
        
        # Generate future periods
        future_x = np.arange(len(historical_data), len(historical_data) + periods)
        future_dates = pd.period_range(
            start=historical_data.index[-1] + 1,
            periods=periods,
            freq='M'
        )
        
        # Calculate forecast with confidence intervals
        forecast_values = trend(future_x)
        std_error = np.std(y - trend(x))
        
        forecast_data = pd.DataFrame({
            'forecast': forecast_values,
            'upper': forecast_values + 1.96 * std_error,
            'lower': forecast_values - 1.96 * std_error
        }, index=future_dates)
        
        return forecast_data
    
    def _create_performance_gauge(self, data: pd.DataFrame) -> None:
        """Create performance gauge chart"""
        st.subheader("⚡ Performance")
        
        kpis = self._calculate_kpis(data)
        savings_rate = kpis['savings_rate']
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=savings_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Savings Rate (%)"},
            delta={'reference': 20, 'increasing': {'color': self.color_palette['success']}},
            gauge={
                'axis': {'range': [-50, 50]},
                'bar': {'color': self.color_palette['primary']},
                'steps': [
                    {'range': [-50, 0], 'color': self.color_palette['danger']},
                    {'range': [0, 20], 'color': self.color_palette['warning']},
                    {'range': [20, 50], 'color': self.color_palette['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_allocation_donut(self, data: pd.DataFrame) -> None:
        """Create allocation donut chart"""
        st.subheader("🎯 Allocation")
        
        if 'Category' not in data.columns or data.empty:
            st.warning("No category data available")
            return
        
        expense_data = data[data['Withdrawls'] > 0]
        category_allocation = expense_data.groupby('Category')['Withdrawls'].sum().sort_values(ascending=False)
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=category_allocation.index,
            values=category_allocation.values,
            hole=0.5,
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Expense Allocation",
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_risk_metrics(self, data: pd.DataFrame) -> None:
        """Create risk metrics display"""
        st.subheader("⚠️ Risk Metrics")
        
        if data.empty:
            st.warning("No data available for risk analysis")
            return
        
        # Calculate risk metrics
        monthly_data = self._prepare_monthly_data(data)
        
        if monthly_data.empty or len(monthly_data) < 2:
            st.warning("Insufficient data for risk analysis")
            return
        
        volatility = monthly_data['net_flow'].std()
        max_drawdown = (monthly_data['net_flow'].cumsum().expanding().max() - monthly_data['net_flow'].cumsum()).max()
        
        # Display metrics
        st.metric("Volatility", f"${volatility:.2f}")
        st.metric("Max Drawdown", f"${max_drawdown:.2f}")
        
        # Risk indicator
        risk_level = "Low" if volatility < 1000 else "Medium" if volatility < 3000 else "High"
        risk_color = self.color_palette['success'] if risk_level == "Low" else self.color_palette['warning'] if risk_level == "Medium" else self.color_palette['danger']
        
        st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    def _apply_portfolio_filters(self, data: pd.DataFrame, date_range: Tuple, categories: List[str]) -> pd.DataFrame:
        """Apply portfolio view filters to data"""
        filtered_data = data.copy()
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (filtered_data['Date'].dt.date >= start_date) &
                (filtered_data['Date'].dt.date <= end_date)
            ]
        
        # Apply category filter
        if categories and 'Category' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Category'].isin(categories)]
        
        return filtered_data

# Global visualization engine instance
_viz_engine: Optional[AdvancedVisualizationEngine] = None

def get_visualization_engine() -> AdvancedVisualizationEngine:
    """Get visualization engine instance (singleton)"""
    global _viz_engine
    
    if _viz_engine is None:
        _viz_engine = AdvancedVisualizationEngine()
    
    return _viz_engine 