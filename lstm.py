import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

from data_handle import data_fetching, split_data
from exponential_moving_average import EMA
from simple_moving_average import SMA
from utils import plot_ema_vs_mid, cal_mse

# constant
window_size=100

# Fetch data
data_fetching("AAPL",start="2010-01-01", end="2022-12-31")

# Read data from csv -> Split data into train and test
df = pd.read_csv('data/stock_data.csv')
train_data, test_data = split_data(df)

N=train_data.size

# Calculate the EMA with a window size of 100
ema = EMA(train_data, window_size)

# Concat all the mid prices
all_mid_data = np.concatenate([train_data,test_data],axis=0)

# Calculate MSE
# mse = cal_mse(pd.DataFrame(train_data), ema)
# print('MSE for EMA: %.5f'%mse[0])

# # Plot EMA predictions vs mid price
# plot_ema_vs_mid(ema, all_mid_data)

# Calculte the SMA
sma = SMA(train_data, window_size)

# Calculate MSE
# mse = cal_mse(pd.DataFrame(train_data), sma)
# print('MSE for SMA: %.5f'%mse[0])

# # Plot EMA predictions vs mid price
# plot_ema_vs_mid(sma, all_mid_data)


# Reshape the data for the LSTM model
# X_train = train_data[:-1]
# y_train = train_data[1:,3]
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# X_test = test_data[:-1]
# y_test = test_data[1:,3]
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Train the LSTM model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# # Plot the training and validation loss
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

