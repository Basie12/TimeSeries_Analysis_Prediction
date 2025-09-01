# engine.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.ML_Pipeline.config import DATA_FILEPATH, DATE_COL_NAME, PROCESSED_DATA_FILEPATH, MODEL_BASE_PATH

# Import from Featureengineering.py (which handles dataset processing internally)
from src.ML_Pipeline.Featureengineering import feature_engineering

# Import from regression_models.py
from src.ML_Pipeline.regression_models import regression, model_dict

# Import from arima.py
from src.ML_Pipeline.arima import arima_model, tune_arima_order

def run_pipeline(input_filepath=DATA_FILEPATH, date_col_name=DATE_COL_NAME, processed_filepath=PROCESSED_DATA_FILEPATH):
    # Run feature engineering (returns df_final; also saves to processed_filepath if needed)
    df = feature_engineering(input_filepath, date_col_name, processed_filepath)
    
    # Prepare data for modeling
    X = df.drop(columns=['avg_weekly_sales'])
    y = df['avg_weekly_sales']
    
    # Split data (consider time-based split for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate each model
    results = {}
    for model_name in model_dict.keys():
        r2, mae, mse = regression(X_train, X_test, y_train, y_test, model_name)
        results[model_name] = {'R2': r2, 'MAE': mae, 'MSE': mse}
        print(f"{model_name} - R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # For ARIMA on global weekly mean
    df_daily = pd.read_csv(input_filepath)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['year'] = df_daily['date'].dt.year
    # Filter out 2025 if present
    df_daily = df_daily[df_daily['year'] != 2025]
    
    # Aggregate to weekly level per SKU
    df_weekly = df_daily.set_index('date').groupby(['sku', pd.Grouper(freq='W')]).agg({
        'units_sold': 'mean',  # Target: avg_weekly_sales
        'price_unit': 'mean',
        'promotion_flag': 'mean',
        'delivery_days': 'mean',
        'stock_available': 'mean',
        'delivered_qty': 'sum',
        'brand': 'first',
        'segment': 'first',
        'category': 'first',
        'channel': lambda x: x.mode()[0] if not x.empty else np.nan,
        'region': lambda x: x.mode()[0] if not x.empty else np.nan,
        'pack_type': lambda x: x.mode()[0] if not x.empty else np.nan
    }).reset_index()
    
    df_weekly.rename(columns={'units_sold': 'avg_weekly_sales'}, inplace=True)
    
    df_weekly.set_index('date', inplace=True)
    df_global = df_weekly[['avg_weekly_sales']]
    df_global = df_global.resample('W').mean()
    
    best_order = (1, 0, 0)
    
    arima_r2, arima_mae, arima_mse, _, _ = arima_model(df_global['avg_weekly_sales'].values, order=best_order)
    results['arima'] = {'R2': arima_r2, 'MAE': arima_mae, 'MSE': arima_mse}
    print(f"ARIMA - R2: {arima_r2:.4f}, MAE: {arima_mae:.4f}, MSE: {arima_mse:.4f}")
    
    # Compare MSE to find best model
    mse_dict = {model: metrics['MSE'] for model, metrics in results.items()}
    best_model = min(mse_dict, key=mse_dict.get)
    print('Best model is {} having MSE of {}'.format(best_model, min(mse_dict.values())))
    
    return results

if __name__ == "__main__":
    run_pipeline()