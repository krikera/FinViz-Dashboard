# FinViz-Dashboard
A Streamlit-based web application for personal finance analytics. Upload CSV financial data to visualize cash flow, track budgets, analyze spending patterns, and detect anomalies. Features interactive charts, custom filters, and predictive insights.

## 🚀 LATEST REFINEMENTS & IMPROVEMENTS

The FinViz Dashboard has been significantly enhanced with modern UI/UX improvements, better error handling, and a more refined user experience:

### ✨ New UI/UX Features
- **Modern Gradient Header**: Professional gradient design with status indicators
- **Enhanced File Upload**: Interactive upload with sample format display and comprehensive help
- **Progress Tracking**: Real-time progress bars during data processing
- **Improved Error Handling**: User-friendly error messages with troubleshooting guides
- **Responsive Design**: Better mobile and tablet compatibility
- **Interactive Elements**: Hover effects and smooth transitions throughout

### 🛠️ Technical Improvements
- **Modular Architecture**: Better code organization with component-based design
- **Enhanced Error Management**: Comprehensive try-catch blocks with graceful degradation
- **Performance Optimizations**: Efficient data processing and lazy loading
- **Type Safety**: Added type hints and improved documentation
- **Logging System**: Comprehensive logging for debugging and monitoring

### 🎯 User Experience Enhancements
- **Sample Data**: Users can view and download sample CSV format
- **Upload Help**: Comprehensive guidance for file preparation
- **Data Validation**: Smart validation with helpful error messages
- **Processing Summary**: Detailed feedback on data processing results
- **Troubleshooting**: Step-by-step problem resolution guides

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
12. [Contact](#contact)

## Overview
The Finance Analytics Dashboard is a powerful, user-friendly web application built with Streamlit that empowers users to gain deep insights into their financial data. By leveraging advanced data visualization techniques and machine learning algorithms, this dashboard provides a comprehensive view of personal or business finances, enabling better financial decision-making.

## ✅ Completed Features

### Core Financial Analytics
- **Data Upload**: Easy CSV file upload functionality for financial data
- **Interactive Data Filtering**: Advanced filtering with date ranges, categories, amounts, and search
- **Cash Flow Analysis**: Visualize monthly cash inflows and outflows with trend analysis
- **Budget Tracking**: Set, track, and monitor budgets for different expense categories
- **Spending Pattern Analysis**: Identify trends and patterns in spending behaviors
- **Anomaly Detection**: Machine learning-based detection of unusual transactions
- **Predictive Analytics**: Forecast future spending using linear regression models
- **Data Export**: Export filtered data in CSV, Excel, or JSON formats with custom formatting

### 🧠 Advanced NLP Features
- **Smart Transaction Categorization**: AI-powered categorization using multiple NLP techniques
- **Sentiment Analysis**: Analyze the emotional context of transaction descriptions
- **Natural Language Search**: Search transactions using plain English queries
- **Word Cloud Visualization**: Visual representation of most common transaction terms
- **Financial Insights Extraction**: Automatically detect merchants, locations, and financial terms
- **Topic Modeling**: Discover spending themes and patterns using advanced algorithms
- **Semantic Similarity**: Find similar transactions using sentence embeddings

### 🎛️ User Experience & Customization
- **Personalized User Settings**: Save preferences for budgets, chart types, and dashboard widgets
- **Customizable Dashboard**: Select and arrange widgets based on individual preferences
- **Quick Actions**: Fast access to data refresh, insights generation, and trend analysis
- **Advanced Search Options**: Multiple search modes including keywords, smart search, and regex
- **Responsive Design**: Optimized interface that works on different screen sizes
- **Error Handling**: Graceful fallback when advanced features are unavailable

### 🔧 Technical Robustness
- **Fallback Mechanisms**: Application works even when advanced NLP libraries are unavailable
- **Runtime Dependency Testing**: Automatically tests and adapts to available libraries
- **Modular Architecture**: Clean separation of concerns across different modules
- **Comprehensive Error Handling**: User-friendly error messages and recovery options

## Technical Stack

### Core Technologies
- **Python 3.8+**: Core programming language
- **Streamlit**: Modern web application framework
- **Pandas**: Advanced data manipulation and analysis
- **NumPy**: Numerical computing foundation
- **Scikit-learn**: Machine learning algorithms and preprocessing

### Visualization Libraries
- **Matplotlib**: Static plotting and charts
- **Plotly**: Interactive and dynamic visualizations
- **Seaborn**: Statistical data visualization
- **WordCloud**: Text visualization for transaction analysis

### NLP & Advanced Analytics
- **TextBlob**: Text processing and sentiment analysis (baseline)
- **SpaCy**: Advanced natural language processing (optional)
- **Transformers**: State-of-the-art NLP models (optional)
- **Sentence Transformers**: Semantic similarity and embeddings (optional)
- **Gensim**: Topic modeling and document similarity (optional)
- **NLTK**: Natural language toolkit (optional)

### Data Processing
- **Dateutil**: Advanced date parsing and manipulation
- **JSON**: Data serialization for user preferences
- **Base64**: File encoding for export functionality

## Installation

### Quick Start (Minimal Dependencies)
For basic functionality without advanced NLP features:

```bash
# Clone the repository
git clone https://github.com/yourusername/finviz-dashboard.git
cd finviz-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal requirements
pip install -r requirements_minimal.txt

# Run the application
streamlit run main.py
```

### Full Installation (All Features)
For complete functionality including advanced NLP:

```bash
# Install all requirements
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm

# Run the application
streamlit run main.py
```

### Docker Installation (Coming Soon)
```bash
docker build -t finviz-dashboard .
docker run -p 8501:8501 finviz-dashboard
```

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

## 📁 Project Structure

### Core Application Files
- `main.py`: Main Streamlit application with integrated user interface and navigation
- `userinput.py`: Comprehensive user input handling and preference management system
- `nlp_engine.py`: Advanced NLP processing engine with fallback mechanisms
- `nlp_visualizations.py`: Interactive NLP analysis dashboard with multiple visualization tabs

### Data Processing & Analytics
- `data_processing.py`: Data preparation, cleaning, and exploratory data analysis
- `utils.py`: Enhanced utility functions with NLP integration
- `model.py`: Machine learning models for anomaly detection and predictions
- `visualization.py`: Core visualization functions for charts and graphs

### Configuration & Documentation
- `requirements.txt`: Complete list of Python dependencies (full installation)
- `requirements_minimal.txt`: Minimal dependencies for basic functionality
- `README.md`: Comprehensive project documentation
- `sample.csv`: Example financial data for testing
- `user_preferences.json`: Auto-generated user settings file

### Key Features by File

#### `main.py` - Main Application
- Streamlit UI orchestration
- File upload and data processing pipeline
- Widget selection and dashboard customization
- Integrated NLP and traditional analytics
- Export functionality with multiple format support

#### `nlp_engine.py` - NLP Core
- Multi-library NLP support with graceful fallbacks
- Advanced transaction categorization
- Sentiment analysis with confidence scoring
- Natural language search capabilities
- Financial insight extraction
- Topic modeling and clustering

#### `nlp_visualizations.py` - NLP Dashboard
- Interactive sentiment analysis charts
- Smart categorization visualizations
- Word cloud generation
- Financial insights and pattern discovery
- Anomaly detection visualization

#### `userinput.py` - User Management
- Persistent user preference storage
- Advanced filtering interface
- Customizable dashboard widgets
- Export configuration options
- Quick action buttons

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

## 🐛 Troubleshooting

### Common Installation Issues

#### TensorFlow/Transformers Import Errors
If you encounter TensorFlow-related errors when starting the application:
```bash
# Use minimal requirements to avoid TensorFlow dependencies
pip install -r requirements_minimal.txt
streamlit run main.py
```
The application will automatically fall back to basic NLP features using TextBlob.

#### SpaCy Model Missing
```bash
# Install SpaCy language model
python -m spacy download en_core_web_sm
```

#### Missing Dependencies
```bash
# Install specific missing packages
pip install textblob nltk wordcloud
```

### Data and Usage Issues

#### CSV Upload Problems
- **Format Issues**: Ensure your CSV has columns: Date, Description, Deposits, Withdrawls, Balance
- **Date Format**: Use YYYY-MM-DD format or common date formats (MM/DD/YYYY, DD/MM/YYYY)
- **Encoding Issues**: Save your CSV in UTF-8 encoding
- **Special Characters**: Remove or escape special characters in descriptions

#### Performance Optimization
- **Large Datasets**: Use date range filters to process smaller chunks
- **Memory Issues**: Disable NLP features for very large datasets (>100k transactions)
- **Slow Loading**: Start with minimal requirements and enable NLP features gradually

#### NLP Features Not Working
- **Check Dependencies**: Verify advanced NLP libraries are installed
- **Fallback Mode**: Application will use basic text processing if advanced libraries unavailable
- **Error Messages**: Check the console for specific library loading errors

### Application Behavior

#### Features Automatically Disabled
The application includes intelligent fallback mechanisms:
- **Advanced NLP → TextBlob**: When transformers/spacy unavailable
- **Semantic Search → Keyword Search**: When sentence transformers unavailable  
- **Topic Modeling → Basic Categorization**: When gensim unavailable
- **Word Clouds → Text Analysis**: When wordcloud library unavailable

#### User Preferences Not Saving
- Check write permissions in the application directory
- Ensure `user_preferences.json` can be created/modified
- Restart the application if preferences appear corrupted

### Getting Help

#### Enable Debug Mode
```bash
streamlit run main.py --logger.level debug
```

#### Check Library Availability
The application displays which NLP libraries are available in the console output when starting.

#### Common Solutions
1. **Restart the application** - Fixes most temporary issues
2. **Clear browser cache** - Resolves Streamlit display issues  
3. **Update dependencies** - `pip install -r requirements.txt --upgrade`
4. **Use minimal setup** - Start with `requirements_minimal.txt` and add features gradually

## Contributing
Contributions to the Finance Analytics Dashboard are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Contact
For any queries or suggestions, please open an issue on the GitHub repository.
