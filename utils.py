from dateutil import parser as date_parser

def custom_date_parser(date_str):
    if date_str:
        try:
            parsed_date = date_parser.parse(date_str)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"Invalid date: {date_str}")
            return None
    else:
        print("Empty date string")
        return None

def categorize_transaction(description):
    categories = {
        'Cheque': ['cheque'],
        'Interest': ['interest'],
        'Tax': ['tax'],
        'Debit Card': ['debit card'],
        'ATM': ['atm'],
        'RTGS': ['rtgs'],
        'Reversal': ['reversal'],
        'Purchase': ['purchase'],
        'Commission': ['commission'],
        'Miscellaneous': ['miscellaneous'],
        'Cash': ['cash'],
        'IMPS': ['imps'],
        'NEFT': ['neft'],
    }
    for category, keywords in categories.items():
        if any(keyword in description.lower() for keyword in keywords):
            return category
    return 'Other'

def identify_recurring_transactions(data):
    data['Recurring'] = data.duplicated(subset=['Description', 'Deposits', 'Withdrawls'], keep=False)
    return data