# arima.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA  # Updated to current ARIMA class
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def estimate_d(series, alpha=0.05, max_d=5):
    """Estimate differencing order (d) using ADF test."""
    d = 0
    current_series = np.asarray(series)
    while d < max_d:
        result = adfuller(current_series)
        p_value = result[1]
        if p_value < alpha:
            return d
        current_series = np.diff(current_series)
        d += 1
    return d

def evaluate_arima_model(X, arima_order):
    """Evaluate ARIMA model with walking forward validation."""
    X = np.asarray(X)
    if len(X) < 10 or np.any(np.isnan(X)):
        return float("inf")
    train_size = int(len(X) * 0.66)
    train, test = X[:train_size], X[train_size:]
    history = list(train)
    predictions = list()
    for t in range(len(test)):
        try:
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()  # Removed disp=0
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        except:
            return float("inf")
    return mean_squared_error(test, predictions)

def tune_arima_order(dataset, p_values=range(0, 3), d_values=range(0, 3), q_values=range(0, 3)):
    """Grid search for best ARIMA order based on MSE."""
    dataset = np.asarray(dataset).astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print(f'ARIMA{order} MSE=%.3f' % mse)
                except Exception as e:
                    print(f'ARIMA{order} failed: {str(e)}')
                    continue
    print(f'Best ARIMA{best_cfg} MSE=%.3f' % best_score)
    return best_cfg

def arima_model(data, order=None, train_size=0.66):
    """Fit and evaluate ARIMA model; auto-tune order if None."""
    X = np.asarray(data).astype('float32')
    if order is None:
        print("Tuning ARIMA order...")
        order = tune_arima_order(X)
        if order is None:
            raise ValueError("Failed to find valid ARIMA order.")
    size = int(len(X) * train_size)
    train, test = X[:size], X[size:]
    history = list(train)
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()  # Removed disp=0
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    return r2, mae, mse, order, predictions