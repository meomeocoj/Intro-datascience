import numpy as np
import pandas as pd
import os
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from data_handle import data_fetching 
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from utils import plot_predictions_vs_true

def preprocessing_data():
    # Fetch data
    if not os.path.exists("data/stock_data.csv"):
        data_fetching('AAPL',start='2010-01-01', end='2022-12-31')
    # read csv
    df = pd.read_csv('data/stock_data.csv')

    # Get the target of the train processs 
    df['Mid Average'] = df[['High','Low']].astype(float).mean(axis=1)

    # Feature engineering 
    window = 50 

    df["MA_50"] = df['Adj Close'].rolling(window=window).mean() # Moving average
    std = df['Adj Close'].rolling(window=window).std()

    df["Upper Band"] = df['MA_50'] + 2*std #Bollinger Band
    df["Lower Band"] = df['MA_50'] - 2*std

    df['Returns'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=window).std() # Volatility

    df['Daily Returns'] = df['Adj Close'].pct_change() # Daily Return

    # Technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close']).rsi() # RSI

    df['MACD'] = ta.trend.MACD(df['Adj Close']).macd() # MACD

    # train_test_split 
    df_train, df_test = train_test_split(df,test_size=0.1,shuffle = False)

    # Mid average price col num
    mid_average_col_num = df.columns.get_loc('Mid Average') - 1
    print(df_train)

    # Normalization
    scale = MinMaxScaler(feature_range=(0,1))
    scale_train_data = scale.fit_transform(df_train.drop(columns='Date'))
    scale_test_data = scale.transform(df_test.drop(columns='Date'))

    # Incorporating timesteps into data
    X_train, y_train = incorporate_timesteps(scale_train_data, target=mid_average_col_num, window_size = window)
    X_test, y_test = incorporate_timesteps(scale_test_data, target=mid_average_col_num, window_size = window)
    return X_train, y_train, X_test, y_test


def incorporate_timesteps(data, target,  window_size):
    X_train = []
    y_train = []
    for i in range(window_size, len(data)):
        X_train.append(data[i - window_size:i,0])
        y_train.append(data[i, target])
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

def run_lstm(X_train, y_train):
    model = create_model(X_train)
    model.fit(X_train,y_train, epochs=25,batch_size=32)
    model.save_weights('./model/lstm')

X_train, y_train, X_test, y_test = preprocessing_data()


# run_lstm(X_train, y_train)


model = create_model(X_train)
model.load_weights('./model/lstm')
result = model.predict(X_test,)
rmse = mean_squared_error(result, y_test,squared = False) 
mape = mean_absolute_percentage_error(y_test, result)
print(mape)
print(rmse)
plot_predictions_vs_true(result, y_test, title="LTSM prediction")





