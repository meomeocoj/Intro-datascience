import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_true(predictions, mid_data, title):
    fig1=plt.figure(figsize=(12, 6))
    plt.plot(mid_data,color='b',label='True')
    plt.plot(predictions,color='orange',label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.title(title)
    plt.legend(fontsize=18)
    plt.show()
