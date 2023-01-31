import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from data_handle import data_fetching, split_data
from sklearn.metrics import mean_squared_error
from utils import plot_predictions_vs_true

def preprocessing_data():
# Fetch data
    if not os.path.exists("data/stock_data.csv"):
        data_fetching('AAPL',start='2010-01-01', end='2022-12-31')
# read csv
    df = pd.read_csv('data/stock_data.csv')
# Split data into train and test
    train_data, test_data = split_data(df)
# Normalization
    scale = MinMaxScaler(feature_range=(0,1))
    scale_train_data = scale.fit_transform(train_data)
    scale_test_data = scale.fit_transform(test_data)
# Incorporating timesteps into data
    window_size = int(scale_train_data.size*0.3)
    X_train, y_train = incorporate_timesteps(scale_train_data, window_size = window_size)
    X_test, y_test = incorporate_timesteps(scale_test_data, window_size = window_size)
    return X_train, y_train, X_test, y_test


def incorporate_timesteps(data,window_size):
    X_train = []
    y_train = []
    for i in range(window_size, data.size):
        X_train.append(data[i - window_size:i,0])
        y_train.append(data[i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def create_model(X_train):
    # Model
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

def run_lstm(X_train):
    model = create_model(X_train)
    history = model.fit(X_train,y_train,epochs=25,batch_size=32)
    model.save_weights('./model/lstm')

X_train, y_train, X_test, y_test = preprocessing_data()
model = create_model(X_train)
model.load_weights('./model/lstm')
result = model.predict(X_test )
rmse = mean_squared_error(result, y_test,squared = False) 
plot_predictions_vs_true(result, y_test, title="LTSM prediction")





