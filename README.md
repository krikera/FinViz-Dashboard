# FinViz-Dashboard
A Streamlit-based web application for personal finance analytics. Upload CSV financial data to visualize cash flow, track budgets, analyze spending patterns, and detect anomalies. Features interactive charts, custom filters, and predictive insights.


# Finance Analytics Dashboard

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technical Stack](#technical-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Requirements](#data-requirements)
7. [Project Structure](#project-structure)
8. [Feature Details](#feature-details)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

## Overview
The Finance Analytics Dashboard is a powerful, user-friendly web application built with Streamlit that empowers users to gain deep insights into their financial data. By leveraging advanced data visualization techniques and machine learning algorithms, this dashboard provides a comprehensive view of personal or business finances, enabling better financial decision-making.

## Features
- **Data Upload**: Easy CSV file upload functionality for financial data.
- **Interactive Data Filtering**: Dynamic filters for date ranges, transaction categories, and amounts.
- **Cash Flow Analysis**: Visualize monthly cash inflows and outflows.
- **Budget Tracking**: Set and track budgets for different expense categories.
- **Spending Pattern Analysis**: Identify trends in spending behaviors.
- **Anomaly Detection**: Automatically detect unusual transactions.
- **Predictive Analytics**: Forecast future spending based on historical data.
- **Data Export**: Export filtered and analyzed data for external use.
- **Responsive Visualizations**: Multiple chart types (Line, Bar, Histogram) for data representation.
- **Summary Statistics**: Quick view of key financial metrics.

## Technical Stack
- **Python 3.7+**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning for anomaly detection and predictions
- **Dateutil**: Advanced date parsing

## Installation
1. Clone the repository:

  &emsp; git clone https://github.com/yourusername/finance-analytics-dashboard.git <br>
  &emsp; cd finance-analytics-dashboard

2. Create a virtual environment (optional but recommended):

&emsp; python -m venv venv <br>
&emsp; source venv/bin/activate  # On Windows use venv\Scripts\activate

3. Install required packages:

&emsp; pip install -r requirements.txt

## Usage
1. Start the Streamlit server :
 streamlit run main.py <br>

2. Open your web browser and navigate to `http://localhost:8501`.

3. Use the sidebar to upload your CSV file and configure dashboard settings.

4. Explore different visualizations and insights provided by the dashboard.

## Data Requirements
Your CSV file should contain the following columns:
- `Date`: Transaction date (format: YYYY-MM-DD)
- `Description`: Transaction description
- `Deposits`: Amount deposited (positive float)
- `Withdrawals`: Amount withdrawn (positive float)
- `Balance`: Account balance after transaction (float)

Example:
Date,Description,Deposits,Withdrawals,Balance
2023-01-01,Salary,5000.00,0.00,5000.00
2023-01-02,Grocery Store,0.00,150.25,4849.75

## Project Structure
- `main.py`: Main application file
- `visualization.py`: Data visualization functions
- `data_processing.py`: Data preparation and processing
- `utils.py`: Utility functions
- `model.py`: Machine learning models
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation (this file)

## Feature Details

### Cash Flow Analysis
Visualizes net cash flow (deposits minus withdrawals) on a monthly basis, helping users understand their overall financial health over time.

### Budget Tracking
Allows users to set budget limits for different expense categories and visualizes actual spending against these limits, highlighting areas where spending exceeds the budget.

### Anomaly Detection
Utilizes the Isolation Forest algorithm to identify unusual transactions based on amount and frequency, helping users spot potential fraudulent activities or unexpected expenses.

### Predictive Analytics
Implements a simple linear regression model to forecast future spending based on historical patterns, aiding in financial planning.

## Customization
Users can customize various aspects of the dashboard:
- Chart types (Line, Bar, Histogram)
- Date range for analysis
- Transaction categories to include/exclude
- Minimum and maximum transaction amounts to consider

## Troubleshooting
- **CSV Upload Issues**: Ensure your CSV file matches the required format. Check for any extra commas or inconsistent date formats.
- **Visualization Errors**: Verify that your data contains valid numerical values for deposits and withdrawals.
- **Performance Issues**: For large datasets, try filtering the date range to a smaller period for faster processing.

## Contributing
Contributions to the Finance Analytics Dashboard are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Contact
For any queries or suggestions, please open an issue on the GitHub repository.
