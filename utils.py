import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_true(ema_predictions, mid_data):
    fig1=plt.figure(figsize=(12, 6))
    plt.plot(mid_data,color='b',label='True')
    plt.plot(ema_predictions,color='orange',label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

def cal_mse(predictions, true):
    return np.square(np.subtract(true, predictions)).mean()