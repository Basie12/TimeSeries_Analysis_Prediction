# lstm.py
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def lstm_model(time_series_data, n_steps=4, epochs=50, batch_size=32, train_size=0.66, model_path=None):
    data = np.asarray(time_series_data).astype('float32').reshape(-1, 1)
    X, y = create_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1)) 
    
    size = int(len(X) * train_size)
    X_train, X_test = X[:size], X[size:]
    y_train, y_test = y[:size], y[size:]
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100 
    
    # Save model
    if model_path is None:
        model_path = 'output/models/lstm_model.h5' 
    model.save(model_path)
    print(f'model saved in {model_path}')
    
    return r2, rmse, mape, y_test, y_pred