import pandas as pd

def EMA(data, window):
    alpha = 2 / (window + 1)
    # train_data to DataFrame
    converted_data = pd.DataFrame(data)
    ema = converted_data.ewm(alpha=alpha, adjust=False).mean()
    return ema