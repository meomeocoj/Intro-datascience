import yfinance as yf
import pandas as pd

def data_fetching(ticker, start, end):
    data = yf.download(ticker, start, end)
    # Clean and preprocess the data
    data.dropna(inplace=True)
    # Write data to csv file
    data.to_csv('data/stock_data.csv')

def split_data(df):
    df['Date'] = df['Date'].str.split(' ').str[0]
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df.sort_values('Date')

    high_prices = df.loc[:,"High"].values
    low_prices = df.loc[:,"Low"].values
    mid_prices = (high_prices + low_prices) / 2.0
    train_data = mid_prices[:int(df.shape[0]*0.8)].reshape(-1,1)
    test_data = mid_prices[int(df.shape[1]*0.8):].reshape(-1,1)

    return (train_data, test_data)


