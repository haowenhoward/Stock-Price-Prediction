"""
Process and visualize stock data using the LSTM model
This script:
1. Imports stock data and saves to CSV
2. Visualizes the raw data
3. Makes predictions using the trained LSTM model
4. Saves predictions to CSV
5. Visualizes the predictions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import os
import shutil

# Configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU is available and configured")
        print("Available GPUs:", physical_devices)
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("No GPU found, using CPU")

# Check GPU availability
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA: ", tf.test.is_built_with_cuda())
print("Is built with GPU: ", tf.test.is_built_with_gpu_support())

from LSTM import StockPredictor, create_dataset, FrequencyCallback

# Set style for better visualizations
sns.set()

def get_stock_data(symbol='MSFT', period='2y'):
    """
    Download stock data and save to CSV
    """
    # Download data
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    # Save raw data
    df.to_csv(f'{symbol}_raw_data.csv')
    
    # Visualize raw data
    plt.figure(figsize=(15, 7))
    plt.plot(df['Close'], label='Stock Price')
    plt.title(f'{symbol} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'{symbol}_raw_data.png')
    plt.close()
    
    return df

def normalize_stock_data(df):
    """
    Normalize stock data using z-score normalization
    Returns the normalized data and the parameters used for normalization
    """
    # Calculate percentage returns
    df['Returns'] = df['Close'].pct_change()
    # Remove first row which will be NaN
    df = df.dropna()
    
    # Normalize returns using z-score
    returns_mean = df['Returns'].mean()
    returns_std = df['Returns'].std()
    clip_std = 3  # Clip at ±3 standard deviations
    
    # Clip and normalize returns
    df['Returns_Normalized'] = np.clip(
        df['Returns'],
        returns_mean - clip_std * returns_std,
        returns_mean + clip_std * returns_std
    )
    df['Returns_Normalized'] = (df['Returns_Normalized'] - returns_mean) / returns_std
    
    return df, returns_mean, returns_std

def reverse_normalize_predictions(predictions_normalized, returns_mean, returns_std, df, timestamp):
    """
    Convert normalized predictions back to actual prices
    """
    # Convert normalized predictions back to returns
    predictions_returns = predictions_normalized * returns_std + returns_mean
    
    # Convert returns back to prices
    last_prices = df['Close'].values[timestamp-1:-1]  # Get the last price before each prediction
    predictions_prices = last_prices * (1 + predictions_returns.flatten())
    
    return predictions_prices, predictions_returns

def train_model(df, timestamp=5, num_layers=1, size_layer=128, dropout_rate=0.2, epochs=300, batch_size=32):
    """
    Train the LSTM model
    Args:
        df: DataFrame containing normalized stock data
        timestamp: Number of days to use for each sequence
        num_layers: Number of LSTM layers
        size_layer: Size of each LSTM layer
        dropout_rate: Dropout rate for regularization
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Create sequences
    X, y = create_dataset(df[['Returns_Normalized']].values, timestamp)
    
    # Initialize and compile model
    model = StockPredictor(num_layers=num_layers, size_layer=size_layer, dropout_rate=dropout_rate)
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    history = model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,
        verbose=0,  # Disable default progress bar
        callbacks=[FrequencyCallback(frequency=10)]  # Show progress every 10 epochs
    )
    
    # Save model and print size
    model_path = 'temp_model.keras'
    model.save(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    print(f"\nModel size: {model_size:.2f} MB")
    # Clean up temporary model
    os.remove(model_path)
    
    return model, history, X, y

def create_dataset(df, timestamp=5):
    """
    Create sequences of data for training with stride=1 to ensure even spacing
    """
    X, y = [], []
    for i in range(0, len(df) - timestamp, 1):  # Use stride=1 for even spacing
        X.append(df[i:(i + timestamp)])
        y.append(df[i + timestamp])
    return np.array(X), np.array(y)

def predict_with_model(model, X, df, returns_mean, returns_std, timestamp):
    """
    Make predictions using a trained model
    """
    # Make predictions
    predictions_normalized = model.predict(X)
    
    # Convert predictions back to prices
    predictions_prices, predictions_returns = reverse_normalize_predictions(
        predictions_normalized, returns_mean, returns_std, df, timestamp
    )
    
    # Create DataFrame with actual and predicted values
    # Note: predictions start after timestamp days
    results_df = pd.DataFrame({
        'Date': df.index[timestamp:],
        'Actual': df['Close'].values[timestamp:],
        'Predicted': predictions_prices,
        'Actual_Returns': df['Returns'].values[timestamp:],
        'Predicted_Returns': predictions_returns.flatten()
    })
    
    # Add NaN values for the first timestamp days to maintain alignment
    padding_df = pd.DataFrame({
        'Date': df.index[:timestamp],
        'Actual': df['Close'].values[:timestamp],
        'Predicted': np.nan,
        'Actual_Returns': df['Returns'].values[:timestamp],
        'Predicted_Returns': np.nan
    })
    
    # Combine padding and results
    results_df = pd.concat([padding_df, results_df])
    
    # Ensure we have predictions for every day after the initial timestamp
    # by interpolating any missing values
    results_df['Predicted'] = results_df['Predicted'].interpolate(method='linear')
    results_df['Predicted_Returns'] = results_df['Predicted_Returns'].interpolate(method='linear')
    
    return results_df

def make_predictions(df, symbol):
    """
    Make predictions using the trained LSTM model
    """
    # Normalize data
    df, returns_mean, returns_std = normalize_stock_data(df)
    
    # Define model configurations
    model_configs = [
        (10, 1, 128, 0.1, 16)
    ]
    
    # Store results
    results = []
    
    # Train models with different configurations
    for i, (timestamp, num_layers, size_layer, dropout_rate, batch_size) in enumerate(model_configs, 1):
        print(f"\nTraining Model {i} with parameters:")
        print(f"timestamp={timestamp}, num_layers={num_layers}, size_layer={size_layer}, dropout_rate={dropout_rate}, batch_size={batch_size}")
        
        # Train model
        model, history, X, y = train_model(
            df, 
            timestamp=timestamp,
            num_layers=num_layers,
            size_layer=size_layer,
            dropout_rate=dropout_rate,
            batch_size=batch_size
        )
        
        # Calculate average loss of last 5 epochs
        average_loss = np.mean(history.history['loss'][-5:])
        results.append({
            'model_id': i,
            'timestamp': timestamp,
            'num_layers': num_layers,
            'size_layer': size_layer,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'average_loss': average_loss
        })
        
        print(f"Average Loss (last 5 epochs): {average_loss:.6f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{symbol}_model_comparison.csv', index=False)
    print(f"\nResults saved to {symbol}_model_comparison.csv")
    
    # Use the best model (lowest loss) for predictions
    best_model_idx = results_df['average_loss'].idxmin()
    best_config = results_df.iloc[best_model_idx]
    print(f"\nUsing best model (Model {best_config['model_id']}) for predictions")
    
    # Train the best model one final time
    model, history, X, y = train_model(
        df,
        timestamp=int(best_config['timestamp']),
        num_layers=int(best_config['num_layers']),
        size_layer=int(best_config['size_layer']),
        dropout_rate=float(best_config['dropout_rate']),
        batch_size=int(best_config['batch_size']),
        epochs=600
    )
    
    # Make predictions with best model
    results_df = predict_with_model(model, X, df, returns_mean, returns_std, int(best_config['timestamp']))
    
    # Limit to last 30 days
    results_df = results_df.tail(30)
    
    # Save predictions
    results_df.to_csv(f'{symbol}_predictions.csv')
    
    # Visualize predictions
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue')
    plt.scatter(results_df['Date'], results_df['Predicted'], label='Predicted', color='red', alpha=0.6)
    plt.title(f'{symbol} Stock Price - Last 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{symbol}_predictions.png')
    plt.close()
    
    # Visualize returns
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Date'], results_df['Actual_Returns'], label='Actual Returns', color='blue')
    plt.scatter(results_df['Date'], results_df['Predicted_Returns'], label='Predicted Returns', color='red', alpha=0.6)
    plt.title(f'{symbol} Daily Returns - Last 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{symbol}_returns_predictions.png')
    plt.close()
    
    return results_df

def main():
    # Define stock symbol
    symbol = 'MSFT'  # Can be changed to any stock symbol
    
    # Get and save raw data
    print(f"Downloading {symbol} stock data...")
    df = get_stock_data(symbol)
    
    # Make and save predictions
    print(f"Making predictions for {symbol}...")
    results = make_predictions(df, symbol)
    
    print(f"Process completed! Check the following files:")
    print(f"1. {symbol}_raw_data.csv - Raw stock data")
    print(f"2. {symbol}_raw_data.png - Raw data visualization")
    print(f"3. {symbol}_predictions.csv - Predictions")
    print(f"4. {symbol}_predictions.png - Predictions visualization")

if __name__ == "__main__":
    main()