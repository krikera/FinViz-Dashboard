from dateutil import parser as date_parser
import pandas as pd
import re
from datetime import datetime

# Initialize NLP engine with robust error handling
nlp_engine = None
try:
    from nlp_engine import FinancialNLPEngine
    nlp_engine = FinancialNLPEngine()
    print("NLP engine initialized successfully")
except Exception as e:
    print(f"NLP engine initialization failed: {e}")
    print("Falling back to basic functionality")
    nlp_engine = None

def custom_date_parser(date_str):
    """Enhanced date parser that handles various financial CSV date formats"""
    if not date_str or pd.isna(date_str):
        return None
    
    # Convert to string if not already
    date_str = str(date_str).strip()
    
    if not date_str:
        return None
    
    try:
        # First try pandas datetime parser (fastest for standard formats)
        return pd.to_datetime(date_str, errors='coerce')
    except:
        pass
    
    try:
        # Try dateutil parser for complex formats
        parsed_date = date_parser.parse(date_str, fuzzy=True)
        return parsed_date
    except:
        pass
    
    # Try common financial CSV formats manually
    date_formats = [
        '%Y-%m-%d',           # 2023-01-01
        '%m/%d/%Y',           # 01/01/2023
        '%d/%m/%Y',           # 01/01/2023 (European)
        '%Y/%m/%d',           # 2023/01/01
        '%d-%m-%Y',           # 01-01-2023
        '%m-%d-%Y',           # 01-01-2023
        '%Y%m%d',             # 20230101
        '%d.%m.%Y',           # 01.01.2023
        '%Y-%m-%d %H:%M:%S',  # 2023-01-01 12:00:00
        '%m/%d/%Y %H:%M:%S',  # 01/01/2023 12:00:00
        '%d/%m/%Y %H:%M:%S',  # 01/01/2023 12:00:00
        '%B %d, %Y',          # January 1, 2023
        '%d %B %Y',           # 1 January 2023
        '%b %d, %Y',          # Jan 1, 2023
        '%d %b %Y',           # 1 Jan 2023
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract date parts using regex
    try:
        # Look for patterns like DD/MM/YYYY, MM/DD/YYYY, etc.
        date_pattern = r'(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,4})'
        match = re.search(date_pattern, date_str)
        
        if match:
            part1, part2, part3 = match.groups()
            
            # Determine which part is year
            if len(part1) == 4:  # YYYY/MM/DD
                year, month, day = int(part1), int(part2), int(part3)
            elif len(part3) == 4:  # MM/DD/YYYY or DD/MM/YYYY
                year = int(part3)
                # Try MM/DD/YYYY first
                try:
                    month, day = int(part1), int(part2)
                    if month > 12:  # Must be DD/MM/YYYY
                        month, day = int(part2), int(part1)
                except:
                    month, day = int(part2), int(part1)
            else:
                # Assume YY format
                if int(part1) > 31:  # Likely year first
                    year = 2000 + int(part1) if int(part1) < 50 else 1900 + int(part1)
                    month, day = int(part2), int(part3)
                else:
                    year = 2000 + int(part3) if int(part3) < 50 else 1900 + int(part3)
                    month, day = int(part1), int(part2)
            
            return datetime(year, month, day)
    except:
        pass
    
    print(f"Warning: Could not parse date '{date_str}', using current date")
    return datetime.now()

def safe_csv_reader(uploaded_file):
    """Enhanced CSV reader with multiple encoding and format attempts"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    separators = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for separator in separators:
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Try reading with different parameters
                df = pd.read_csv(
                    uploaded_file, 
                    encoding=encoding, 
                    sep=separator,
                    thousands=',',
                    decimal='.',
                    parse_dates=False,  # We'll handle dates manually
                    low_memory=False
                )
                
                # Check if we have minimum required columns
                required_cols = ['Date', 'Description']
                if all(col in df.columns or any(col.lower() in c.lower() for c in df.columns) for col in required_cols):
                    print(f"Successfully read CSV with encoding: {encoding}, separator: '{separator}'")
                    return df, encoding, separator
                    
            except Exception as e:
                continue
    
    # If all attempts failed, try with default pandas settings
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, low_memory=False, parse_dates=False)
        return df, 'utf-8', ','
    except Exception as e:
        raise ValueError(f"Could not read CSV file with any encoding or separator. Error: {str(e)}")

def normalize_column_names(df):
    """Normalize column names to standard format"""
    # Mapping of common variations to standard names
    column_mapping = {
        # Date variations
        'date': 'Date',
        'transaction date': 'Date',
        'trans date': 'Date',
        'posting date': 'Date',
        'value date': 'Date',
        'dated': 'Date',
        
        # Description variations
        'description': 'Description',
        'memo': 'Description',
        'details': 'Description',
        'transaction details': 'Description',
        'narration': 'Description',
        'reference': 'Description',
        'particulars': 'Description',
        
        # Amount variations
        'amount': 'Amount',
        'debit': 'Withdrawls',
        'credit': 'Deposits',
        'withdrawal': 'Withdrawls',
        'deposit': 'Deposits',
        'withdrawls': 'Withdrawls',
        'deposits': 'Deposits',
        'debit amount': 'Withdrawls',
        'credit amount': 'Deposits',
        'withdrawal amount': 'Withdrawls',
        'deposit amount': 'Deposits',
        
        # Balance variations
        'balance': 'Balance',
        'running balance': 'Balance',
        'available balance': 'Balance',
        'closing balance': 'Balance',
    }
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Normalize column names
    new_columns = {}
    for col in df_copy.columns:
        col_lower = col.lower().strip()
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]
        else:
            # Keep original column name
            new_columns[col] = col
    
    df_copy = df_copy.rename(columns=new_columns)
    
    # If we have a single 'Amount' column, split into Deposits and Withdrawls
    if 'Amount' in df_copy.columns and 'Deposits' not in df_copy.columns and 'Withdrawls' not in df_copy.columns:
        df_copy['Deposits'] = df_copy['Amount'].apply(lambda x: x if pd.notnull(x) and x > 0 else 0)
        df_copy['Withdrawls'] = df_copy['Amount'].apply(lambda x: abs(x) if pd.notnull(x) and x < 0 else 0)
    
    return df_copy

def categorize_transaction(description):
    """Enhanced categorization using NLP if available, fallback to rule-based"""
    # Try NLP categorization first
    if nlp_engine and nlp_engine.nlp_available:
        try:
            nlp_results = nlp_engine.advanced_transaction_categorization([description])
            if nlp_results and nlp_results[0]['confidence'] > 0.3:
                return nlp_results[0]['category']
        except Exception as e:
            print(f"NLP categorization failed: {e}")
    
    # Fallback to original rule-based categorization
    categories = {
        'food_dining': ['restaurant', 'cafe', 'food', 'dining', 'pizza', 'burger', 'coffee', 'lunch', 'dinner', 'starbucks', 'mcdonalds', 'kfc'],
        'shopping': ['store', 'shop', 'mall', 'amazon', 'purchase', 'buy', 'retail', 'walmart', 'target', 'costco'],
        'transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'metro', 'bus', 'parking', 'toll', 'vehicle'],
        'utilities': ['electric', 'water', 'internet', 'phone', 'cable', 'utility', 'power', 'gas_bill'],
        'healthcare': ['hospital', 'doctor', 'medical', 'pharmacy', 'health', 'dental', 'clinic', 'medicine'],
        'entertainment': ['movie', 'cinema', 'theater', 'netflix', 'spotify', 'game', 'entertainment', 'music'],
        'finance': ['bank', 'atm', 'interest', 'fee', 'loan', 'credit', 'investment', 'insurance'],
        'education': ['school', 'university', 'education', 'tuition', 'book', 'course', 'training'],
        'travel': ['hotel', 'flight', 'travel', 'booking', 'airbnb', 'vacation', 'trip', 'airline'],
        'other': ['cheque', 'interest', 'tax', 'debit card', 'atm', 'rtgs', 'reversal', 'commission', 'miscellaneous', 'cash', 'imps', 'neft']
    }
    
    description_lower = str(description).lower()
    for category, keywords in categories.items():
        if any(keyword in description_lower for keyword in keywords):
            return category
    return 'other'

def identify_recurring_transactions(data):
    """Enhanced recurring transaction identification with NLP insights"""
    data['Recurring'] = data.duplicated(subset=['Description', 'Deposits', 'Withdrawls'], keep=False)
    
    # Add NLP-based recurring pattern detection
    if nlp_engine and nlp_engine.nlp_available:
        try:
            descriptions = data['Description'].tolist()
            insights = nlp_engine.extract_financial_insights(descriptions)
            
            # Add merchant information if available
            if insights.get('merchants'):
                data['Detected_Merchants'] = data['Description'].apply(
                    lambda x: [merchant for merchant in insights['merchants'] if merchant.lower() in x.lower()]
                )
            
        except Exception as e:
            print(f"NLP insights extraction failed: {e}")
    
    return data

def enhanced_transaction_analysis(descriptions):
    """Provide comprehensive NLP analysis of transactions"""
    if not nlp_engine or not nlp_engine.nlp_available:
        return None
    
    try:
        analysis = {
            'categorization': nlp_engine.advanced_transaction_categorization(descriptions),
            'sentiment': nlp_engine.sentiment_analysis(descriptions),
            'insights': nlp_engine.extract_financial_insights(descriptions),
            'anomalies': nlp_engine.detect_anomalous_descriptions(descriptions)
        }
        return analysis
    except Exception as e:
        print(f"Enhanced analysis failed: {e}")
        return None

def smart_search_transactions(data, query):
    """Natural language search through transactions"""
    if not nlp_engine or not nlp_engine.nlp_available:
        # Fallback to simple text search
        return data[data['Description'].str.contains(query, case=False, na=False)]
    
    try:
        descriptions = data['Description'].tolist()
        return nlp_engine.natural_language_search(descriptions, query, data)
    except Exception as e:
        print(f"Smart search failed: {e}")
        return data[data['Description'].str.contains(query, case=False, na=False)]

def generate_insights_summary(data):
    """Generate NLP-powered insights summary"""
    if not nlp_engine or not nlp_engine.nlp_available:
        return "NLP insights not available"
    
    try:
        return nlp_engine.generate_spending_summary(data)
    except Exception as e:
        return f"Insights generation failed: {e}"