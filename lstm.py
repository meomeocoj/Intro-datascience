import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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

# Reshape the data for the LSTM model
X_train = train_data[:-1]
y_train = train_data[1:,3]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = test_data[:-1]
y_test = test_data[1:,3]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Plot the training and validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

