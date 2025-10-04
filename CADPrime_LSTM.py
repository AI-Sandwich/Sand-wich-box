#%% Load Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping

#%% Step 1: Fetch Canadian Yield Curve Data (from Bank of Canada)
def fetch_yield_curve_data(path):
    
    yield_curve = pd.read_csv(path, index_col=0, parse_dates=True)
    yield_curve.columns = [col.replace(' ', '') for col in yield_curve.columns]
    yield_curve = yield_curve[[col for col in yield_curve.columns if not col=='']]
    yield_curve = yield_curve.map(lambda x: str(x).replace(' ', '') if isinstance(x, str) else x).replace(['na',''], np.nan).dropna()

    
    return yield_curve

#%% Step 2: Fetch Canadian Prime Rate (from Yahoo Finance)
def fetch_prime_rate():
    # Bank of Canada policy rate is a good proxy for prime rate
    boc_rate = yf.download('^IRX',start='1990-01-01')['Close'] / 100  # Convert to decimal
    prime_rate = boc_rate + 0.015  # Prime is typically policy rate + 1.5%
    return prime_rate.resample('D').ffill()

#%% Step 3: Combine and prepare data
def prepare_data(yield_curve, prime_rate):
    # Merge datasets
    data = pd.concat([yield_curve, prime_rate], axis=1)
    data.columns = list(yield_curve.columns) + ['PrimeRate']
    data = data.replace([' '], np.nan).dropna()
    axis = data.index
    
    # Normalize data

    scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data)


    
    return scaled_data, scaler, axis

#%% Step 4: Create time series samples
def create_dataset(data, lookback=60, forecast_horizon=30):
    X, y = [], []
    for i in range(len(data)-lookback-forecast_horizon):
        X.append(data[i:(i+lookback)])
        y.append(data[(i+lookback):(i+lookback+forecast_horizon), -1])  # Only predict PrimeRate
    return np.array(X), np.array(y)

#%% Step 5: Build Encoder-Decoder LSTM model
def build_model(lookback, n_features, forecast_horizon):
    # Encoder
    encoder_inputs = Input(shape=(lookback, n_features))
    encoder = LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # Decoder
    decoder_inputs = RepeatVector(forecast_horizon)(encoder_outputs)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = TimeDistributed(Dense(1))
    outputs = decoder_dense(decoder_outputs)
    
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

#%% Main execution
if __name__ == "__main__":
    # Fetch data
    
    yield_curve = fetch_yield_curve_data(r"C:\Users\gosly\Downloads\yield_curves.csv")
    prime_rate = fetch_prime_rate()
    
    # Prepare data
    data, scaler, axis = prepare_data(yield_curve, prime_rate)
    
    # Create training set
    lookback = 60  # 3 months of daily data
    forecast_horizon = 30  # Predict 1 month ahead
    X, y = create_dataset(data, lookback, forecast_horizon)
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = build_model(lookback, X.shape[2], forecast_horizon)
    history = model.fit(X_train, y_train, 
                       epochs=100, 
                       batch_size=32, 
                       validation_data=(X_test, y_test),
                       callbacks=[EarlyStopping(patience=10)])
    
    # Evaluate
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    dummy_data = np.zeros((len(predictions), forecast_horizon, data.shape[1]))
    dummy_data[:, :, -1] = predictions.squeeze()
    predictions = scaler.inverse_transform(dummy_data.reshape(-1, data.shape[1]))[:, -1]
    predictions = predictions.reshape(len(predictions)//forecast_horizon, forecast_horizon)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(axis, 
             scaler.inverse_transform(data)[:, -1], 
             label='Actual')
    
    plt.plot(axis[-len(y_test)*forecast_horizon:], 
             predictions[:,0], 
             label='Predicted', alpha=0.7)
    plt.title('Canadian Prime Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Prime Rate')
    plt.legend()
    plt.show()
# %%
