import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Load data using yfinance
def load_data(crypto_currency, against_currency, start='2015-01-01', end=None):
    if end is None:
        end = pd.Timestamp.now()
    df = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)
    return df

# Prepare training data
def prepare_training_data(df, prediction_days=60, future_days=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_days):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

# Prepare test data
def prepare_test_data(df, prediction_days=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    model_inputs = scaled_data[-(len(df) + prediction_days):]
    x_test, actual_prices = [], df['Close'].values

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, actual_prices, scaler
