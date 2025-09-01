# models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from .config import PROCESSED_DATA_FILEPATH, MODEL_BASE_PATH

model_dict = {
    'lin_reg': LinearRegression,
    'rf_reg': RandomForestRegressor,
    'xgb_reg': XGBRegressor
}

def regression(X_train, X_test, y_train, y_test, model_name, model_path=None):
    if model_name not in model_dict:
        raise ValueError(f'Only these options for model_name are allowed: {list(model_dict.keys())}')
    
    # Use global if model_path not provided
    if model_path is None:
        model_path = f"{MODEL_BASE_PATH}{model_name}_model.pkl"
    
    # No scaler needed since data is pre-scaled
    model = model_dict[model_name]()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)  # Save just the model
    print(f'model saved in {model_path}')
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_FILEPATH)
    
    # Assuming 'avg_weekly_sales' is the target
    X = df.drop(columns=['avg_weekly_sales'])
    y = df['avg_weekly_sales']
    
    # Split data (consider time-based split for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate each model
    for model_name in model_dict.keys():
        r2, mae, mse = regression(X_train, X_test, y_train, y_test, model_name)
        print(f"{model_name} - R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")