import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf
import os

from data_handle import data_fetching, split_data
from exponential_moving_average import EMA
from simple_moving_average import SMA
from utils import plot_predictions_vs_true, cal_mse

start = "2010-01-01"
end = "2022-12-31"
window_size = 100

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker:", "AAPL")
df = yf.download(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

# Fetch data
if not os.path.exists("data/stock_data.csv"): 
    data_fetching('AAPL',start='2010-01-01', end='2022-12-31')
# Read csv
df = pd.read_csv('data/stock_data.csv')
# Split data into train and test
train_data, test_data = split_data(df)
N = train_data.size
# Concat train data and test data for visualization and test
all_mid_data = np.concatenate([train_data,test_data],axis=0)

#  ------ Start -------
print('Starting.....')
# Exponential Moving Average
ema = EMA(all_mid_data, window_size)
# Simple moving average 
sma = SMA(all_mid_data, window_size)
# Cal MSE 
mse1 = cal_mse(ema, all_mid_data)
mse2 = cal_mse(sma, all_mid_data)
print('MSE for EMA: %.5f'%mse1[0])
print('MSE for SMA: %.5f'%mse2[0])
# Plot Predictions vs True
plot_predictions_vs_true(ema, all_mid_data)
plot_predictions_vs_true(sma, all_mid_data)
print('End.')



# Visualizations
# st.subheader('Closing Price vs Time chart')
# fig = plt.figure(figsize = (12,6))
# plt.plot(df.Close)
# st.pyplot(fig)