import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    if len(data) < 10:
        data['Anomaly'] = 1
        return data
    contamination = min(0.05, max(0.01, 5.0 / len(data)))
    clf = IsolationForest(contamination=contamination, random_state=42)
    data['Anomaly'] = clf.fit_predict(data[['Deposits', 'Withdrawls']])
    return data

def train_predict_model(data):
    data = data[['Date', 'Withdrawls']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())
    X = data['DateOrdinal'].values.reshape(-1, 1)
    y = data['Withdrawls'].values

    split_idx = int(len(data) * 0.8)
    if split_idx < 2 or (len(data) - split_idx) < 1:
        return None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)
    results = pd.DataFrame({'DateOrdinal': X_test.flatten(), 'Actual': y_test, 'Predicted': predictions})
    results['Date'] = results['DateOrdinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))
    results = results.drop('DateOrdinal', axis=1)

    return results