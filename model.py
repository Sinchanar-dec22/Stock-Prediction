import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

def engineer_features(df):
    df = df.copy()
    df['MA_7']  = df['Close'].rolling(7).mean()
    df['MA_21'] = df['Close'].rolling(21).mean()
    df['Return'] = df['Close'].pct_change()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_7'] = df['Close'].shift(7)
    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    df = engineer_features(df)

    features = ['MA_7', 'MA_21', 'Return', 'Lag_1', 'Lag_3', 'Lag_7']
    X = df[features].values
    y = df['Close'].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    # Predict tomorrow
    last_row = scaler.transform([df[features].iloc[-1].values])
    tomorrow = model.predict(last_row)[0]

    return y_test, y_pred, rmse, tomorrow, df.index[split:]
