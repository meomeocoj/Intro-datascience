import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_fetching(ticker, start, end):
    data = yf.download(ticker, start, end)
    # Clean and preprocess the data
    data.dropna(inplace=True)
    # Write data to csv file
    data.to_csv('data/stock_data.csv')

def split_data():
    # Load the stock data from a csv file
    df = pd.read_csv('data/stock_data.csv')

    # Drop unnecessary columns
    df.drop(['Adj Close','Date'], axis=1, inplace=True)

    # Convert the data into a numpy array
    data = df.values

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split the data into training and test sets
    train_data = data[:int(data.shape[0]*0.8),:]
    test_data = data[int(data.shape[0]*0.8):,:]

    return (train_data, test_data)

