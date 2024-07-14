import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    clf = IsolationForest(contamination=0.01, random_state=42)
    data['Anomaly'] = clf.fit_predict(data[['Deposits', 'Withdrawls']])
    return data

def train_predict_model(data):
    data = data[['Date', 'Withdrawls']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())
    X = data['DateOrdinal'].values.reshape(-1, 1)
    y = data['Withdrawls'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    results = pd.DataFrame({'DateOrdinal': X_test.flatten(), 'Actual': y_test, 'Predicted': predictions})
    results['Date'] = results['DateOrdinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))
    results = results.drop('DateOrdinal', axis=1)

    return results