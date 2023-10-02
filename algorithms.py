import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib

# Load and preprocess data


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data  # return daily data


# LSTM Model
def lstm_model(filepath):
    data = load_and_preprocess_data(filepath)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train[['Receipt_Count']])
    scaled_test = scaler.transform(test[['Receipt_Count']])
    joblib.dump(scaler, 'lstm_scaler.pkl')

    X_train, y_train = [], []
    for i in range(len(scaled_train) - 1):
        X_train.append(scaled_train[i:(i + 1), 0])
        y_train.append(scaled_train[i + 1, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(30, input_shape=(
        X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(5, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)
    model.save("lstm_model.h5")

    test_predictions = []
    first_eval_batch = scaled_train[-1:]
    current_batch = first_eval_batch.reshape((1, 1, 1))

    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [
                                  [current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(
        np.array(test_predictions).reshape(-1, 1))

    mae = mean_absolute_error(test['Receipt_Count'], true_predictions)
    rmse = np.sqrt(mean_squared_error(test['Receipt_Count'], true_predictions))

    # Save the scaler, MAE, and RMSE
    joblib.dump(mae, 'lstm_mae.pkl')
    joblib.dump(rmse, 'lstm_rmse.pkl')
    return mae, rmse, model, scaler


# Gradient Boosting Model
def gradient_boosting_model(filepath):
    data = load_and_preprocess_data(filepath)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    X_train, y_train = pd.DataFrame(index=train.index), train['Receipt_Count']
    X_test = pd.DataFrame(index=test.index)

    for i in range(1, 4):
        X_train[f'lag_{i}'] = train['Receipt_Count'].shift(i)
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index]

    for i in range(1, 4):
        X_test[f'lag_{i}'] = test['Receipt_Count'].shift(i)

    # Handle cases where X_test might end up being empty after dropping NaN values
    if X_test.dropna().empty:
        raise ValueError(
            "Not enough data points in the test set to create lag features.")

    X_test.dropna(inplace=True)

    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    actuals = test.loc[X_test.index, 'Receipt_Count']
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    return mae, rmse, model


# Exponential Smoothing State Space Model (ETS)
def ets_model(filepath):
    data = load_and_preprocess_data(filepath)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    model = ExponentialSmoothing(np.asarray(
        train['Receipt_Count']), seasonal='add', seasonal_periods=12)
    fit = model.fit()

    predictions = fit.forecast(len(test))
    mae = mean_absolute_error(test['Receipt_Count'], predictions)
    rmse = np.sqrt(mean_squared_error(test['Receipt_Count'], predictions))

    return mae, rmse, fit


def predict_lstm(model, scaler, n_days):
    data = load_and_preprocess_data('data_daily.csv')
    predictions = []
    first_eval_batch = scaler.transform(data[-1:])
    current_batch = first_eval_batch.reshape((1, 1, 1))

    for i in range(n_days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [
            [current_pred]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def predict_gb(model, n_days):
    data = load_and_preprocess_data('data_daily.csv')
    predictions = []

    for i in range(n_days):
        last_known_values = data['Receipt_Count'].tail(3).values.reshape(1, -1)
        prediction = model.predict(last_known_values)[0]
        predictions.append(prediction)
        # Adjusted to days for daily data
        new_index = data.index[-1] + pd.DateOffset(days=1)
        data = data.append(pd.DataFrame(
            {'Receipt_Count': [prediction]}, index=[new_index]))

    return predictions


def predict_ets(model, n_days):
    return model.forecast(steps=n_days)
