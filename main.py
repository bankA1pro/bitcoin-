import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime

# 1. Data Collection
def fetch_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download('BTC-USD', start='2015-01-01', end=end_date)
    return data

# 2. Data Preprocessing
def preprocess_data(data):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Split into train and test sets (80-20)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    # Create sequences (using 60 days for prediction)
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    seq_length = 60
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler, close_prices

# 3. Build and Train LSTM Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Make Prediction
def predict_next_day(model, data, scaler):
    last_60_days = data[-60:].values
    scaled_last_60 = scaler.transform(last_60_days.reshape(-1, 1))
    X_pred = np.reshape(scaled_last_60, (1, 60, 1))
    predicted_price = model.predict(X_pred)
    return scaler.inverse_transform(predicted_price)[0][0]

# Main Execution
if __name__ == "__main__":
    # Fetch and preprocess data
    data = fetch_data()
    X_train, y_train, X_test, y_test, scaler, close_prices = preprocess_data(data)
    
    # Build and train model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
    
    # Predict next day's price
    current_price = close_prices[-1][0]
    predicted_price = predict_next_day(model, close_prices, scaler)
    
    # Determine direction and profit
    direction = "up" if predicted_price > current_price else "down"
    percentage_change = abs((predicted_price / current_price - 1) * 100)
    
    # User input and profit calculation
    investment = float(input("Enter the amount to invest: $"))
    profit = investment * (percentage_change / 100) if direction == "up" else -investment * (percentage_change / 100)
    
    # Output results
    print(f"\nPrediction: Bitcoin price will go {direction}.")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"Percentage Change: {percentage_change:.2f}%")
    print(f"Estimated Profit: ${profit:.2f} ({'Profit' if profit > 0 else 'Loss'})")
