import pandas as pd;

def SMA(data, window):
    return pd.DataFrame(data).rolling(window=window).mean()