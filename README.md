# Stock Price Prediction using LSTM

This project uses a LSTM model (Using Tensorflow Keras API) to predict future stock price. Specifically, it reads in the previous 10 consequtive days of stock prices, and predicts for the next day.

Results Are Displayed in MSFT_predictions.png, a comparison between predicted price and actual price, accross 30 Days.

----------Frameworks Used----------

- Pulling Data using Yh Finance
- Z-score Data Normalization
- LSTM Model
- Data stored using CSV files
- Graph Plotting with Matplotlib and Seaborn
- Python 3.12
- TensorFlow 2.19

----------Trained Using----------

- Geforce RTX 4050
- CUDA 12.5
- CUDNN 9.3
- WSL2 Environment
- 600 Epoches, on 2 years of stock data

----------Model Architecture----------

- LSTM Layer (128 units): 66,560 parameters
- Dense Layer (1 unit): 129 parameters
- Total Parameters: 66,689

Using MIT License