import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


from data_handle import data_fetching

start = "2010-01-01"
end = "2022-12-31"
window_size = 100

def get_target(df):
    high_prices = df.loc[:,"High"].values
    low_prices = df.loc[:,"Low"].values
    mid_prices = (high_prices + low_prices) / 2.0
    df["Mid Price"] = mid_prices
    return df

def computeSMA(data, window):
    sma = data.rolling(window=window).mean()
    return sma

def computeEMA(data, span):
    ema = data.ewm(span=span, adjust=False).mean()
    return ema

def construct_df(df):
    df["Date"] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

    for i in range(50, 250, 50):
        df["SMA_{}".format(i)] = computeSMA(df["Mid Price"], i)
    for i in range(50, 250, 50):
        df["EMA_{}".format(i)] = computeEMA(df["Mid Price"], i)

    return df

def plot_data_SMA(df):
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title("Applec - Price chart (Mid Price) vs SMA")
    ax.set_ylabel("Mid Price ($)")


    monthly_locator = mdates.MonthLocator()
    half_year_locator = mdates.MonthLocator(interval=6)
    month_year_formatter = mdates.DateFormatter('%b, %Y')
    ax.xaxis.set_major_locator(half_year_locator)
    ax.xaxis.set_minor_locator(monthly_locator)
    ax.xaxis.set_major_formatter(month_year_formatter)

    ax.plot(df["Date"], df['Mid Price'], label="True")
    for i in range(50, 250, 50):
        ax.plot(df["Date"], df['SMA_{}'.format(i)], label="SMA_{}".format(i))

    fig.autofmt_xdate()
    plt.legend(fontsize=14)
    plt.show()
    
def plot_data_EMA(df):
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title("Applec - Price chart (Mid Price) vs EMA")
    ax.set_ylabel("Mid Price ($)")


    monthly_locator = mdates.MonthLocator()
    half_year_locator = mdates.MonthLocator(interval=6)
    month_year_formatter = mdates.DateFormatter('%b, %Y')
    ax.xaxis.set_major_locator(half_year_locator)
    ax.xaxis.set_minor_locator(monthly_locator)
    ax.xaxis.set_major_formatter(month_year_formatter)

    ax.plot(df["Date"], df['Mid Price'], label="True")
    for i in range(50, 250, 50):
        ax.plot(df['Date'], df['EMA_{}'.format(i)], label="EMA_{}".format(i))

    fig.autofmt_xdate()
    plt.legend(fontsize=14)
    plt.show() 

def plot_data_EMA_SMA(df, window):
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title("Apple Inc - Price chart (Mid Price) vs EMA vs SMA with window size of 50")
    ax.set_ylabel("Price ($)")


    monthly_locator = mdates.MonthLocator()
    half_year_locator = mdates.MonthLocator(interval=6)
    month_year_formatter = mdates.DateFormatter('%b, %Y')
    ax.xaxis.set_major_locator(half_year_locator)
    ax.xaxis.set_minor_locator(monthly_locator)
    ax.xaxis.set_major_formatter(month_year_formatter)

    ax.plot(df["Date"], df['Mid Price'], label="True")
    ax.plot(df['Date'], df['SMA_{}'.format(window)], label="SMA_{}".format(window))
    ax.plot(df['Date'], df['EMA_{}'.format(window)], label="EMA_{}".format(window))

    fig.autofmt_xdate()
    plt.legend(fontsize=14)
    plt.show()     

# Fetch data
if not os.path.exists("data/stock_data.csv"): 
    data_fetching('AAPL',start='2010-01-01', end='2022-12-31')
# Read csv
df = pd.read_csv('data/stock_data.csv')
df = get_target(df)
df = df.reset_index()
df = construct_df(df)



# Calculate RMSE 
sma_prediction_data = df["SMA_50"][50:]
true_data = df["Mid Price"][50:]
ema_prediction_data = df["EMA_50"][50:]
ema_rmse = mean_squared_error(ema_prediction_data, true_data, squared=False)
sma_rmse = mean_squared_error(sma_prediction_data, true_data, squared=False)

print('RMSE of SMA: %.5f'%sma_rmse)
print('RMSE of EMA: %.5f'%ema_rmse)

# Calculate MAPE
def MAPE(true, predict):
    mape = np.mean(np.abs((true - predict)/true))*100
    return mape

ema_mape = mean_absolute_percentage_error(true_data, ema_prediction_data)*100
sma_mape = mean_absolute_percentage_error(true_data, sma_prediction_data)*100

print('MAPE of SMA: %.5f'%sma_mape)
print('MAPE of EMA: %.5f'%ema_mape)


# Visualization
plot_data_SMA(df)
plot_data_EMA(df)
plot_data_EMA_SMA(df, 50)