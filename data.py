import yfinance as yf

def data_fetching(ticker, start, end):
    data = yf.download(ticker, start, end)
# Clean and preprocess the data
    data.dropna(inplace=True)
    data.to_csv('data/stock_data.csv')

data_fetching("AAPL",start="2010-01-01", end="2022-12-31")
