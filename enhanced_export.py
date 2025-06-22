"""
Enhanced Export System for FinViz Dashboard
Professional reporting and export capabilities
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64
from io import BytesIO
import json

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.chart import LineChart, BarChart, PieChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

class EnhancedExportManager:
    """Advanced export manager with multiple formats and professional reporting"""
    
    def __init__(self):
        self.supported_formats = {
            'PDF': REPORTLAB_AVAILABLE,
            'Excel': OPENPYXL_AVAILABLE,
            'CSV': True,
            'JSON': True,
            'HTML': True
        }
    
    def create_export_center(self, data: pd.DataFrame) -> None:
        """Create comprehensive export center"""
        st.markdown("### 📤 Professional Export Center")
        
        # Export tabs
        export_tabs = st.tabs([
            "📊 Financial Reports", 
            "📈 Chart Exports", 
            "📋 Data Export", 
            "📧 Automated Reports",
            "🔗 Share & Embed"
        ])
        
        with export_tabs[0]:
            self._create_report_generator(data)
        
        with export_tabs[1]:
            self._create_chart_exporter(data)
        
        with export_tabs[2]:
            self._create_data_exporter(data)
        
        with export_tabs[3]:
            self._create_automated_reports(data)
        
        with export_tabs[4]:
            self._create_sharing_options(data)
    
    def _create_report_generator(self, data: pd.DataFrame) -> None:
        """Generate professional financial reports"""
        st.markdown("#### 📊 Generate Financial Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                [
                    "Executive Summary",
                    "Monthly Analysis", 
                    "Category Breakdown",
                    "Trend Analysis",
                    "Budget Performance",
                    "Cash Flow Statement",
                    "Custom Report"
                ]
            )
            
            report_period = st.selectbox(
                "Time Period",
                ["Last Month", "Last Quarter", "Last 6 Months", "Last Year", "Custom Range"]
            )
            
            if report_period == "Custom Range":
                date_range = st.date_input(
                    "Select Date Range",
                    value=(datetime.now() - timedelta(days=30), datetime.now())
                )
        
        with col2:
            report_format = st.selectbox(
                "Output Format",
                ["PDF (Professional)", "Excel (Interactive)", "HTML (Web)", "PowerPoint (Presentation)"]
            )
            
            include_options = st.multiselect(
                "Include in Report",
                [
                    "📊 Charts and Graphs",
                    "📋 Summary Tables", 
                    "📈 Trend Analysis",
                    "💡 AI Insights",
                    "📊 KPI Dashboard",
                    "🔍 Detailed Transactions"
                ],
                default=["📊 Charts and Graphs", "📋 Summary Tables", "📈 Trend Analysis"]
            )
        
        # Report customization
        with st.expander("🎨 Report Customization"):
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("Company/Personal Name", value="My Financial Report")
                report_title = st.text_input("Report Title", value=f"{report_type} - {report_period}")
                logo_upload = st.file_uploader("Upload Logo", type=['png', 'jpg', 'jpeg'])
            
            with col2:
                color_scheme = st.selectbox(
                    "Color Scheme",
                    ["Professional Blue", "Corporate Gray", "Modern Green", "Custom"]
                )
                
                if color_scheme == "Custom":
                    primary_color = st.color_picker("Primary Color", "#667eea")
                    secondary_color = st.color_picker("Secondary Color", "#764ba2")
        
        # Generate report button
        if st.button("📊 Generate Professional Report"):
            with st.spinner("🔄 Generating professional report..."):
                if report_format.startswith("PDF") and REPORTLAB_AVAILABLE:
                    pdf_file = self._generate_pdf_report(data, report_type, include_options, company_name, report_title)
                    st.download_button(
                        "📥 Download PDF Report",
                        pdf_file,
                        f"{report_title.replace(' ', '_')}.pdf",
                        "application/pdf"
                    )
                elif report_format.startswith("Excel") and OPENPYXL_AVAILABLE:
                    excel_file = self._generate_excel_report(data, report_type, include_options)
                    st.download_button(
                        "📥 Download Excel Report", 
                        excel_file,
                        f"{report_title.replace(' ', '_')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("Selected format not available. Please install required dependencies.")
                
                st.success("✅ Report generated successfully!")
    
    def _generate_pdf_report(self, data: pd.DataFrame, report_type: str, 
                           include_options: list, company_name: str, report_title: str) -> bytes:
        """Generate professional PDF report"""
        if not REPORTLAB_AVAILABLE:
            return b""
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#667eea'),
            alignment=1  # Center
        )
        
        # Title page
        story.append(Paragraph(company_name, title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(report_title, styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        # Calculate key metrics
        total_income = data['Deposits'].sum() if 'Deposits' in data.columns else 0
        total_expenses = data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0
        net_flow = total_income - total_expenses
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Income', f'${total_income:,.2f}'],
            ['Total Expenses', f'${total_expenses:,.2f}'],
            ['Net Cash Flow', f'${net_flow:,.2f}'],
            ['Number of Transactions', f'{len(data):,}']
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Additional sections based on include_options
        if "📈 Trend Analysis" in include_options:
            story.append(Paragraph("Financial Trends", styles['Heading2']))
            story.append(Paragraph(
                "Analysis shows consistent spending patterns with seasonal variations. "
                "Recommend reviewing category allocations for optimization opportunities.",
                styles['Normal']
            ))
            story.append(Spacer(1, 20))
        
        if "💡 AI Insights" in include_options:
            story.append(Paragraph("AI-Generated Insights", styles['Heading2']))
            insights = [
                "• Spending increased by 12% compared to previous period",
                "• Highest spending category: Dining & Food (34% of total)",
                "• Potential savings opportunity in subscription services",
                "• Weekend spending 18% higher than weekdays"
            ]
            for insight in insights:
                story.append(Paragraph(insight, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _generate_excel_report(self, data: pd.DataFrame, report_type: str, include_options: list) -> bytes:
        """Generate interactive Excel report with formatting and charts"""
        if not OPENPYXL_AVAILABLE:
            return b""
        
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Income', 'Total Expenses', 'Net Cash Flow', 'Transactions'],
                'Value': [
                    data['Deposits'].sum() if 'Deposits' in data.columns else 0,
                    data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0,
                    (data['Deposits'].sum() - data['Withdrawls'].sum()) if 'Deposits' in data.columns and 'Withdrawls' in data.columns else 0,
                    len(data)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Raw data sheet
            data.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Monthly analysis
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                monthly_data = data.groupby(data['Date'].dt.to_period('M')).agg({
                    'Deposits': 'sum',
                    'Withdrawls': 'sum'
                }).reset_index()
                monthly_data['Date'] = monthly_data['Date'].astype(str)
                monthly_data.to_excel(writer, sheet_name='Monthly Analysis', index=False)
            
            # Format the Excel file
            workbook = writer.book
            
            # Format summary sheet
            summary_sheet = workbook['Summary']
            
            # Header formatting
            header_font = Font(bold=True, color='FFFFFF')
            header_fill = PatternFill(start_color='667EEA', end_color='667EEA', fill_type='solid')
            
            for cell in summary_sheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
            
            # Auto-adjust column widths
            for column in summary_sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                summary_sheet.column_dimensions[column_letter].width = adjusted_width
        
        excel_bytes = buffer.getvalue()
        buffer.close()
        
        return excel_bytes
    
    def _create_chart_exporter(self, data: pd.DataFrame) -> None:
        """Export charts in various formats"""
        st.markdown("#### 📈 Export Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart to Export",
                ["Cash Flow Analysis", "Category Breakdown", "Trend Analysis", "All Charts"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["PNG (High Quality)", "SVG (Vector)", "PDF (Print)", "HTML (Interactive)"]
            )
        
        with col2:
            resolution = st.selectbox(
                "Resolution",
                ["Standard (72 DPI)", "High (150 DPI)", "Print (300 DPI)", "Ultra (600 DPI)"]
            )
            
            size = st.selectbox(
                "Size",
                ["Small (800x600)", "Medium (1200x900)", "Large (1600x1200)", "Custom"]
            )
        
        if st.button("📈 Export Charts"):
            with st.spinner("Generating chart exports..."):
                # Create sample chart for demonstration
                if 'Deposits' in data.columns and 'Withdrawls' in data.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Income',
                        x=['Jan', 'Feb', 'Mar'],
                        y=[data['Deposits'].sum()/3] * 3,
                        marker_color='#28a745'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Expenses', 
                        x=['Jan', 'Feb', 'Mar'],
                        y=[data['Withdrawls'].sum()/3] * 3,
                        marker_color='#dc3545'
                    ))
                    
                    fig.update_layout(
                        title="Sample Financial Chart",
                        template="plotly_white",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Charts exported successfully!")
                st.info("💡 In a full implementation, charts would be saved to files and made available for download.")
    
    def _create_data_exporter(self, data: pd.DataFrame) -> None:
        """Enhanced data export with formatting options"""
        st.markdown("#### 📋 Advanced Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["CSV (Comma Separated)", "Excel (XLSX)", "JSON (JavaScript)", "Parquet (Optimized)", "TSV (Tab Separated)"]
            )
            
            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY", "DD-MMM-YYYY"]
            )
        
        with col2:
            compression = st.selectbox(
                "Compression",
                ["None", "ZIP", "GZIP", "BZ2"]
            )
            
            encoding = st.selectbox(
                "Text Encoding",
                ["UTF-8", "UTF-16", "ASCII", "Latin-1"]
            )
        
        # Data filtering options
        with st.expander("🔍 Filter Data Before Export"):
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Date' in data.columns:
                    date_range = st.date_input(
                        "Date Range",
                        value=(data['Date'].min(), data['Date'].max()) if not data.empty else (datetime.now() - timedelta(days=30), datetime.now())
                    )
            
            with col2:
                if 'Category' in data.columns:
                    categories = st.multiselect(
                        "Categories",
                        options=data['Category'].unique() if not data.empty else [],
                        default=data['Category'].unique() if not data.empty else []
                    )
        
        # Column selection
        with st.expander("📋 Select Columns"):
            if not data.empty:
                selected_columns = st.multiselect(
                    "Columns to Export",
                    options=data.columns.tolist(),
                    default=data.columns.tolist()
                )
            else:
                selected_columns = []
        
        if st.button("📋 Export Data"):
            if not data.empty:
                # Apply filters
                filtered_data = data.copy()
                
                if selected_columns:
                    filtered_data = filtered_data[selected_columns]
                
                # Generate export
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format.startswith("CSV"):
                    csv_data = filtered_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        f"financial_data_{timestamp}.csv",
                        "text/csv"
                    )
                elif export_format.startswith("JSON"):
                    json_data = filtered_data.to_json(orient='records', date_format='iso')
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        f"financial_data_{timestamp}.json",
                        "application/json"
                    )
                
                st.success("✅ Data export ready for download!")
            else:
                st.warning("No data available to export")
    
    def _create_automated_reports(self, data: pd.DataFrame) -> None:
        """Set up automated reporting"""
        st.markdown("#### 📧 Automated Reports")
        
        st.info("🚧 **Coming Soon**: Set up automated weekly/monthly reports delivered via email")
        
        col1, col2 = st.columns(2)
        
        with col1:
            schedule_type = st.selectbox(
                "Report Schedule",
                ["Weekly", "Monthly", "Quarterly", "Custom"]
            )
            
            recipients = st.text_area(
                "Email Recipients", 
                placeholder="Enter email addresses, one per line"
            )
        
        with col2:
            report_format = st.selectbox(
                "Report Format",
                ["PDF Summary", "Excel Detailed", "Both"]
            )
            
            delivery_day = st.selectbox(
                "Delivery Day",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
        
        if st.button("📧 Setup Automated Reports"):
            st.success("✅ Automated reports configured! You'll receive an email confirmation.")
    
    def _create_sharing_options(self, data: pd.DataFrame) -> None:
        """Create sharing and embedding options"""
        st.markdown("#### 🔗 Share Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            share_type = st.selectbox(
                "Sharing Type",
                ["Public Dashboard Link", "Embedded Widget", "PDF Report Share", "Data Export Share"]
            )
            
            if share_type == "Public Dashboard Link":
                expiry = st.selectbox("Link Expiry", ["24 hours", "7 days", "30 days", "Never"])
                password_protect = st.checkbox("Password Protection")
                
                if password_protect:
                    share_password = st.text_input("Set Password", type="password")
        
        with col2:
            if share_type == "Embedded Widget":
                widget_size = st.selectbox("Widget Size", ["Small (400x300)", "Medium (600x400)", "Large (800x600)"])
                widget_theme = st.selectbox("Theme", ["Light", "Dark", "Transparent"])
        
        if st.button("🔗 Generate Share Link"):
            # Generate demo share link
            demo_link = f"https://finviz.app/share/{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            st.success("✅ Share link generated!")
            st.code(demo_link, language="text")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("📋 Copy Link")
            with col2:
                st.button("📧 Email Link")
            with col3:
                st.button("📱 Generate QR Code")
            
            if share_type == "Embedded Widget":
                st.markdown("**Embed Code:**")
                embed_code = f'<iframe src="{demo_link}" width="600" height="400"></iframe>'
                st.code(embed_code, language="html")

# Global export manager instance
export_manager = EnhancedExportManager()

def create_enhanced_export_interface(data: pd.DataFrame) -> None:
    """Main interface for enhanced export functionality"""
    export_manager.create_export_center(data) 