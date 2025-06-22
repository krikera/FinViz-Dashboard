import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any
import logging
from data_processing import track_budget

# Configure logging
logger = logging.getLogger(__name__)

def visualize_data(data, column, title, chart_type):
    """Legacy visualization function - enhanced with Plotly"""
    try:
        if data.empty or column not in data.columns:
            st.warning(f"No data available for {column}")
            return
        
        # Use Plotly for better interactivity
        if chart_type == 'Line Chart':
            fig = px.line(data, x='Date', y=column, title=title)
            fig.update_traces(line=dict(width=3))
        elif chart_type == 'Bar Chart':
            fig = px.bar(data, x='Date', y=column, title=title)
        elif chart_type == 'Histogram':
            fig = px.histogram(data, x=column, title=title, nbins=30)
        else:
            # Fallback to matplotlib for unsupported types
            plt.figure(figsize=(12, 6))
            if chart_type == 'Line Chart':
                plt.plot(data['Date'], data[column], marker='o', linewidth=2)
            elif chart_type == 'Bar Chart':
                plt.bar(data['Date'], data[column])
            elif chart_type == 'Histogram':
                plt.hist(data[column], bins=30, alpha=0.7)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(column, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
            return
        
        # Configure Plotly chart
        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in visualize_data: {e}")
        st.error(f"Error creating visualization: {str(e)}")

def visualize_cash_flow(data, chart_type):
    """Enhanced cash flow visualization with multiple perspectives"""
    try:
        st.markdown("### 💰 Cash Flow Analysis")
        
        # Prepare data
        data['Date'] = pd.to_datetime(data['Date'])
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        # Group by YearMonth and sum numeric columns
        cash_flow = data.groupby('YearMonth').agg({
            'Deposits': 'sum',
            'Withdrawls': 'sum'
        }).reset_index()
        
        cash_flow['Net Flow'] = cash_flow['Deposits'] - cash_flow['Withdrawls']
        cash_flow['Month'] = cash_flow['YearMonth'].dt.to_timestamp()
        cash_flow['Month_str'] = cash_flow['Month'].dt.strftime('%Y-%m')
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Monthly Overview", "📈 Trends", "🔍 Detailed Analysis"])
        
        with tab1:
            render_monthly_cash_flow(cash_flow)
        
        with tab2:
            render_cash_flow_trends(cash_flow)
        
        with tab3:
            render_detailed_cash_flow_analysis(data, cash_flow)
            
    except Exception as e:
        logger.error(f"Error in visualize_cash_flow: {e}")
        st.error(f"Error creating cash flow visualization: {str(e)}")

def render_monthly_cash_flow(cash_flow):
    """Render monthly cash flow overview"""
    try:
        # Main cash flow chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Cash Flow', 'Cumulative Net Flow'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Monthly bars
        fig.add_trace(
            go.Bar(
                x=cash_flow['Month_str'],
                y=cash_flow['Deposits'],
                name='Income',
                marker_color='#28a745',
                hovertemplate='Income: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=cash_flow['Month_str'],
                y=-cash_flow['Withdrawls'],
                name='Expenses',
                marker_color='#dc3545',
                hovertemplate='Expenses: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Net flow line
        fig.add_trace(
            go.Scatter(
                x=cash_flow['Month_str'],
                y=cash_flow['Net Flow'],
                mode='lines+markers',
                name='Net Flow',
                line=dict(color='#007bff', width=3),
                marker=dict(size=8),
                hovertemplate='Net Flow: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Cumulative net flow
        cumulative_flow = cash_flow['Net Flow'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=cash_flow['Month_str'],
                y=cumulative_flow,
                mode='lines+markers',
                name='Cumulative Flow',
                line=dict(color='#17a2b8', width=3),
                marker=dict(size=6),
                fill='tonexty' if cumulative_flow.iloc[-1] > 0 else 'tozeroy',
                fillcolor='rgba(23, 162, 184, 0.1)',
                hovertemplate='Cumulative: $%{y:,.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template="plotly_white",
            title_text="Monthly Cash Flow Analysis",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Amount ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_income = cash_flow['Deposits'].mean()
            st.metric("Avg Monthly Income", f"${avg_income:,.2f}")
        
        with col2:
            avg_expenses = cash_flow['Withdrawls'].mean()
            st.metric("Avg Monthly Expenses", f"${avg_expenses:,.2f}")
        
        with col3:
            avg_net = cash_flow['Net Flow'].mean()
            delta_color = "normal" if avg_net >= 0 else "inverse"
            st.metric("Avg Net Flow", f"${avg_net:,.2f}", delta=f"${avg_net:,.2f}", delta_color=delta_color)
        
        with col4:
            total_net = cash_flow['Net Flow'].sum()
            st.metric("Total Net Flow", f"${total_net:,.2f}")
            
    except Exception as e:
        logger.error(f"Error in render_monthly_cash_flow: {e}")
        st.error("Error rendering monthly cash flow")

def render_cash_flow_trends(cash_flow):
    """Render cash flow trends and patterns"""
    try:
        # Trend analysis
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Income vs Expenses Trend', 'Net Flow Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Income vs Expenses
        fig.add_trace(
            go.Scatter(
                x=cash_flow['Month_str'],
                y=cash_flow['Deposits'],
                mode='lines+markers',
                name='Income',
                line=dict(color='#28a745', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cash_flow['Month_str'],
                y=cash_flow['Withdrawls'],
                mode='lines+markers',
                name='Expenses',
                line=dict(color='#dc3545', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Net flow with trend line
        fig.add_trace(
            go.Scatter(
                x=cash_flow['Month_str'],
                y=cash_flow['Net Flow'],
                mode='lines+markers',
                name='Net Flow',
                line=dict(color='#007bff', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Add trend line for net flow
        if len(cash_flow) > 1:
            x_numeric = np.arange(len(cash_flow))
            trend = np.polyfit(x_numeric, cash_flow['Net Flow'], 1)
            trend_line = np.poly1d(trend)(x_numeric)
            
            fig.add_trace(
                go.Scatter(
                    x=cash_flow['Month_str'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#ff6b35', dash='dash', width=2)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth analysis
        if len(cash_flow) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                income_growth = ((cash_flow['Deposits'].iloc[-1] - cash_flow['Deposits'].iloc[0]) / cash_flow['Deposits'].iloc[0] * 100) if cash_flow['Deposits'].iloc[0] != 0 else 0
                st.metric("Income Growth", f"{income_growth:.1f}%")
            
            with col2:
                expense_growth = ((cash_flow['Withdrawls'].iloc[-1] - cash_flow['Withdrawls'].iloc[0]) / cash_flow['Withdrawls'].iloc[0] * 100) if cash_flow['Withdrawls'].iloc[0] != 0 else 0
                st.metric("Expense Growth", f"{expense_growth:.1f}%")
                
    except Exception as e:
        logger.error(f"Error in render_cash_flow_trends: {e}")
        st.error("Error rendering cash flow trends")

def render_detailed_cash_flow_analysis(data, cash_flow):
    """Render detailed cash flow analysis"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily transaction volume
            daily_stats = data.groupby('Date').agg({
                'Deposits': 'sum',
                'Withdrawls': 'sum'
            }).reset_index()
            daily_stats['Transaction_Count'] = data.groupby('Date').size().values
            
            fig_volume = px.scatter(
                daily_stats,
                x='Date',
                y='Transaction_Count',
                size='Withdrawls',
                color='Deposits',
                title="Daily Transaction Activity",
                labels={'Transaction_Count': 'Number of Transactions'},
                color_continuous_scale='Viridis'
            )
            
            fig_volume.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Cash flow volatility
            cash_flow['Volatility'] = cash_flow['Net Flow'].rolling(window=3, min_periods=1).std()
            
            fig_vol = px.line(
                cash_flow,
                x='Month_str',
                y='Volatility',
                title="Cash Flow Volatility",
                labels={'Volatility': 'Standard Deviation ($)'}
            )
            
            fig_vol.update_traces(line=dict(color='#ff6b35', width=3))
            fig_vol.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Monthly breakdown table
        st.markdown("#### 📊 Monthly Breakdown")
        summary_table = cash_flow[['Month_str', 'Deposits', 'Withdrawls', 'Net Flow']].copy()
        summary_table.columns = ['Month', 'Income ($)', 'Expenses ($)', 'Net Flow ($)']
        summary_table['Income ($)'] = summary_table['Income ($)'].apply(lambda x: f"{x:,.2f}")
        summary_table['Expenses ($)'] = summary_table['Expenses ($)'].apply(lambda x: f"{x:,.2f}")
        summary_table['Net Flow ($)'] = summary_table['Net Flow ($)'].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(summary_table, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in render_detailed_cash_flow_analysis: {e}")
        st.error("Error rendering detailed analysis")

def visualize_budget_tracking(data, chart_type, budget_limits):
    """Enhanced budget tracking visualization"""
    try:
        st.markdown("### 📊 Budget Tracking")
        
        if not budget_limits:
            st.info("💡 Set budget limits to track your spending against targets")
            return
        
        # Process budget data
        data = track_budget(data, budget_limits)
        
        # Create budget analysis
        budget_analysis = data.groupby('Category').agg({
            'Withdrawls': 'sum',
            'Budget Limit': 'first',
            'Over Budget': 'first'
        }).reset_index()
        
        budget_analysis['Remaining'] = budget_analysis['Budget Limit'] - budget_analysis['Withdrawls']
        budget_analysis['Usage %'] = (budget_analysis['Withdrawls'] / budget_analysis['Budget Limit'] * 100).round(1)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            render_budget_overview(budget_analysis)
        
        with col2:
            render_budget_progress_bars(budget_analysis)
        
        # Detailed budget table
        render_budget_details_table(budget_analysis)
        
    except Exception as e:
        logger.error(f"Error in visualize_budget_tracking: {e}")
        st.error(f"Error creating budget visualization: {str(e)}")

def render_budget_overview(budget_analysis):
    """Render budget overview chart"""
    try:
        # Budget vs Actual spending
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Budget',
            x=budget_analysis['Category'],
            y=budget_analysis['Budget Limit'],
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Actual',
            x=budget_analysis['Category'],
            y=budget_analysis['Withdrawls'],
            marker_color=['red' if over else 'green' for over in budget_analysis['Over Budget']]
        ))
        
        fig.update_layout(
            title="Budget vs Actual Spending",
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in render_budget_overview: {e}")
        st.error("Error rendering budget overview")

def render_budget_progress_bars(budget_analysis):
    """Render budget progress as horizontal bars"""
    try:
        fig = go.Figure()
        
        # Create horizontal bar chart
        colors = ['red' if usage > 100 else 'orange' if usage > 80 else 'green' 
                 for usage in budget_analysis['Usage %']]
        
        fig.add_trace(go.Bar(
            y=budget_analysis['Category'],
            x=budget_analysis['Usage %'],
            orientation='h',
            marker_color=colors,
            text=[f"{usage:.1f}%" for usage in budget_analysis['Usage %']],
            textposition='inside'
        ))
        
        # Add 100% reference line
        fig.add_vline(x=100, line_dash="dash", line_color="red", opacity=0.7)
        
        fig.update_layout(
            title="Budget Usage Percentage",
            xaxis_title="Usage %",
            yaxis_title="Category",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in render_budget_progress_bars: {e}")
        st.error("Error rendering budget progress")

def render_budget_details_table(budget_analysis):
    """Render detailed budget breakdown table"""
    try:
        st.markdown("#### 📋 Budget Details")
        
        # Prepare display data
        display_data = budget_analysis.copy()
        display_data['Budget Limit'] = display_data['Budget Limit'].apply(lambda x: f"${x:,.2f}")
        display_data['Withdrawls'] = display_data['Withdrawls'].apply(lambda x: f"${x:,.2f}")
        display_data['Remaining'] = display_data['Remaining'].apply(lambda x: f"${x:,.2f}")
        display_data['Usage %'] = display_data['Usage %'].apply(lambda x: f"{x:.1f}%")
        display_data['Status'] = display_data['Over Budget'].apply(lambda x: "🔴 Over Budget" if x else "✅ Within Budget")
        
        # Select and rename columns for display
        display_columns = ['Category', 'Budget Limit', 'Withdrawls', 'Remaining', 'Usage %', 'Status']
        display_data = display_data[display_columns]
        display_data.columns = ['Category', 'Budget ($)', 'Spent ($)', 'Remaining ($)', 'Usage', 'Status']
        
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary metrics
        over_budget_count = budget_analysis['Over Budget'].sum()
        total_categories = len(budget_analysis)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Categories Tracked", total_categories)
        with col2:
            st.metric("Over Budget", over_budget_count)
        with col3:
            compliance_rate = ((total_categories - over_budget_count) / total_categories * 100) if total_categories > 0 else 0
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
            
    except Exception as e:
        logger.error(f"Error in render_budget_details_table: {e}")
        st.error("Error rendering budget details")

def visualize_spending_patterns_and_predictions(data, chart_type):
    """Enhanced spending patterns and predictions visualization"""
    try:
        st.markdown("### 📈 Spending Patterns & Predictions")
        
        from model import train_predict_model
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📊 Current Patterns", "🔮 Predictions", "📅 Seasonal Analysis"])
        
        with tab1:
            render_spending_patterns(data, chart_type)
        
        with tab2:
            render_predictions(data)
        
        with tab3:
            render_seasonal_analysis(data)
            
    except Exception as e:
        logger.error(f"Error in visualize_spending_patterns_and_predictions: {e}")
        st.error(f"Error creating spending patterns visualization: {str(e)}")

def render_spending_patterns(data, chart_type):
    """Render current spending patterns"""
    try:
        # Daily spending pattern
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.day_name()
        data['Hour'] = data['Date'].dt.hour
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by day of week
            daily_spending = data.groupby('DayOfWeek')['Withdrawls'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_daily = px.bar(
                x=daily_spending.index,
                y=daily_spending.values,
                title="Average Spending by Day of Week",
                labels={'x': 'Day', 'y': 'Average Amount ($)'},
                color=daily_spending.values,
                color_continuous_scale='Blues'
            )
            
            fig_daily.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Transaction size distribution
            non_zero_withdrawals = data[data['Withdrawls'] > 0]['Withdrawls']
            
            if not non_zero_withdrawals.empty:
                fig_dist = px.histogram(
                    x=non_zero_withdrawals,
                    title="Transaction Size Distribution",
                    labels={'x': 'Transaction Amount ($)', 'y': 'Frequency'},
                    nbins=20
                )
                
                fig_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Spending heatmap by day and category
        if 'Category' in data.columns:
            category_day_spending = data.groupby(['DayOfWeek', 'Category'])['Withdrawls'].sum().reset_index()
            category_pivot = category_day_spending.pivot(index='DayOfWeek', columns='Category', values='Withdrawls').fillna(0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            category_pivot = category_pivot.reindex(day_order)
            
            fig_heatmap = px.imshow(
                category_pivot.values,
                x=category_pivot.columns,
                y=category_pivot.index,
                title="Spending Heatmap: Category vs Day of Week",
                color_continuous_scale='Blues',
                aspect='auto'
            )
            
            fig_heatmap.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
    except Exception as e:
        logger.error(f"Error in render_spending_patterns: {e}")
        st.error("Error rendering spending patterns")

def render_predictions(data):
    """Render spending predictions"""
    try:
        from model import train_predict_model
        
        # Generate predictions
        predictions = train_predict_model(data)
        
        if predictions is not None and not predictions.empty:
            # Combine historical and predicted data
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Withdrawls'],
                mode='lines+markers',
                name='Historical Spending',
                line=dict(color='#007bff', width=2),
                marker=dict(size=4)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted'],
                mode='lines+markers',
                name='Predicted Spending',
                line=dict(color='#ff6b35', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Spending Predictions",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            if len(predictions) > 0:
                avg_predicted = predictions['Predicted'].mean()
                current_avg = data['Withdrawls'].mean()
                change_pct = ((avg_predicted - current_avg) / current_avg * 100) if current_avg > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Avg Spending", f"${current_avg:.2f}")
                with col2:
                    st.metric("Predicted Avg Spending", f"${avg_predicted:.2f}")
                with col3:
                    st.metric("Expected Change", f"{change_pct:+.1f}%")
        else:
            st.warning("Unable to generate predictions. More data may be needed.")
            
    except Exception as e:
        logger.error(f"Error in render_predictions: {e}")
        st.error("Error generating predictions")

def render_seasonal_analysis(data):
    """Render seasonal spending analysis"""
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data['Quarter'] = data['Date'].dt.quarter
        data['MonthName'] = data['Date'].dt.month_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly spending pattern
            monthly_spending = data.groupby('MonthName')['Withdrawls'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            fig_monthly = px.line(
                x=monthly_spending.index,
                y=monthly_spending.values,
                title="Average Monthly Spending Pattern",
                labels={'x': 'Month', 'y': 'Average Amount ($)'},
                markers=True
            )
            
            fig_monthly.update_layout(template="plotly_white", height=400)
            fig_monthly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Quarterly spending
            quarterly_spending = data.groupby('Quarter')['Withdrawls'].sum()
            quarter_labels = [f"Q{q}" for q in quarterly_spending.index]
            
            fig_quarterly = px.bar(
                x=quarter_labels,
                y=quarterly_spending.values,
                title="Quarterly Spending Distribution",
                labels={'x': 'Quarter', 'y': 'Total Amount ($)'},
                color=quarterly_spending.values,
                color_continuous_scale='Blues'
            )
            
            fig_quarterly.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Seasonal insights
        if len(monthly_spending.dropna()) > 0:
            highest_month = monthly_spending.idxmax()
            lowest_month = monthly_spending.idxmin()
            
            st.info(f"💡 **Seasonal Insights:** Highest spending typically occurs in {highest_month}, while {lowest_month} tends to have the lowest spending.")
            
    except Exception as e:
        logger.error(f"Error in render_seasonal_analysis: {e}")
        st.error("Error rendering seasonal analysis")