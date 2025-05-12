"""
Stock Price Prediction using LSTM (Long Short-Term Memory) Neural Network
This script implements a deep learning model to predict stock prices using historical data.
The model uses TensorFlow 2.x with Keras for time series forecasting.
"""

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm

# Set seaborn style for better visualizations
sns.set()

# Download stock data
def download_stock_data(symbol='GOOG', period='1y'):
    """
    Download stock data using yfinance
    Args:
        symbol (str): Stock symbol (default: 'GOOG')
        period (str): Data period to download (default: '1y')
    Returns:
        pandas.DataFrame: Historical stock data
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

# Load and preprocess the data
df = download_stock_data()
minmax = MinMaxScaler().fit(df[['Close']].astype('float32'))
df_log = minmax.transform(df[['Close']].astype('float32'))
df_log = pd.DataFrame(df_log)

# Define training parameters
test_size = 30
simulation_size = 10
df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]

class StockPredictor(tf.keras.Model):
    """
    LSTM Model implementation for stock price prediction using Keras
    """
    def __init__(self, num_layers, size_layer, dropout_rate=0.2):
        super(StockPredictor, self).__init__()
        self.lstm_layers = [
            tf.keras.layers.LSTM(size_layer, 
                               return_sequences=(i < num_layers - 1),
                               dropout=dropout_rate)
            for i in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return self.dense(x)

class FrequencyCallback(tf.keras.callbacks.Callback):
    """
    Callback to control the frequency of training output
    """
    def __init__(self, frequency=10):
        super(FrequencyCallback, self).__init__()
        self.frequency = frequency
        self.epochs_seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_seen += 1
        if self.epochs_seen % self.frequency == 0:
            print(f'\nEpoch {self.epochs_seen}/{self.params["epochs"]} - loss: {logs["loss"]:.4f} - val_loss: {logs["val_loss"]:.4f}')

def create_dataset(df, timestamp=5):
    """
    Create sequences of data for training
    """
    X, y = [], []
    for i in range(len(df) - timestamp):
        X.append(df[i:(i + timestamp)])
        y.append(df[i + timestamp])
    return np.array(X), np.array(y)

def calculate_accuracy(real, predict):
    """
    Calculate the accuracy of predictions
    """
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def anchor(signal, weight):
    """
    Smooth the signal using exponential moving average
    """
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

# Model parameters
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 0.2
learning_rate = 0.01

def forecast():
    """
    Train the LSTM model and forecast future stock prices
    """
    # Prepare training data
    X_train, y_train = create_dataset(df_train.values, timestamp)
    X_test, y_test = create_dataset(df_test.values, timestamp)
    
    # Create and compile model
    model = StockPredictor(num_layers, size_layer, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epoch,
        batch_size=32,
        validation_split=0.1,
        verbose=0,  # Disable default progress bar
        callbacks=[FrequencyCallback(frequency=10)]  # Show progress every 10 epochs
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    return minmax.inverse_transform(predictions)

def run_simulations():
    """
    Run multiple simulations and plot the results
    """
    # Run multiple simulations
    results = []
    for i in range(simulation_size):
        print('simulation %d' % (i + 1))
        results.append(forecast())

    # Calculate accuracy for each simulation
    accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

    # Plot results
    plt.figure(figsize=(15, 5))
    for no, r in enumerate(results):
        plt.plot(r, label='forecast %d' % (no + 1))
    plt.plot(df['Close'].iloc[-test_size:].values, label='true trend', c='black')
    plt.legend()
    plt.title('average accuracy: %.4f' % (np.mean(accuracies)))
    plt.show()

if __name__ == "__main__":
    run_simulations()




