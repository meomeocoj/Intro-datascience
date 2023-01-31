import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_lstm_model(df):
    # Calculate RSI
    rsi = talib.RSI(df['close'], timeperiod=14)
    df['rsi'] = rsi
    
    # Normalize RSI values to the range [0, 1]
    min_rsi = df['rsi'].min()
    max_rsi = df['rsi'].max()
    df['rsi_norm'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)
    
    # Normalize all the features to the range [0, 1]
    scaler = MinMaxScaler()
    df_norm = scaler.fit_transform(df[['high', 'low', 'close', 'open', 'rsi']])
    df_norm = pd.DataFrame(df_norm, columns=['high_norm', 'low_norm', 'close_norm', 'open_norm', 'rsi_norm'])
    df = pd.concat([df, df_norm], axis=1)
    
    # Define the number of timesteps
    timesteps = 10
    
    # Split the data into input sequences
    X = []
    y = []
    for i in range(timesteps, df.shape[0]):
        X.append(df[i-timesteps:i][['high_norm', 'low_norm', 'close_norm', 'open_norm', 'rsi_norm']].values)
        y.append(df.iloc[i]['close_tomorrow_norm'])
    X = np.array(X)
    y = np.array(y)
    
    # Split the data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = tf.keras.metrics.MeanSquaredError()
   
