import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def data_fetching(ticker, start, end):
    data = yf.download(ticker, start, end)
# Clean and preprocess the data
    data.dropna(inplace=True)
# Feature engineering
    data["14d_ma"] = data["Close"].rolling(window=14).mean()
    data["14d_std"] = data["Close"].rolling(window=14).std()
    data["14d_bollinger_high"] = data["14d_ma"] + (data["14d_std"] * 2)
    data["14d_bollinger_low"] = data["14d_ma"] - (data["14d_std"] * 2)
# Split data into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    return (train, test)


# Collect data using yfinance API
ticker = "AAPL"
train, test = data_fetching(ticker, start="2010-01-01", end="2022-12-31")
X_train = train.drop(["Close"], axis=1)
y_train = train["Close"]
X_test = test.drop(["Close"], axis=1)
y_test = test["Close"]

print(X_test)


# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Make predictions
new_data= data_fetching(ticker, start="2023-01-01", end="2023-01-31")
new_X = new_data[0].drop(["Close"], axis=1)
new_Y = new_data[0]['Close']
new_y_pred = model.predict(new_X)
mae = mean_absolute_error(new_y_pred, new_Y)
print("Mean Absolute Error:", mae)
