# engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.ML_Pipeline.config import DATA_FILEPATH, DATE_COL_NAME, PROCESSED_DATA_FILEPATH, MODEL_BASE_PATH

# Import from Featureengineering.py (which handles dataset processing internally)
from src.ML_Pipeline.Featureengineering import feature_engineering

# Import from regression_models.py
from src.ML_Pipeline.regression_models import regression, model_dict

# Import from arima.py
from src.ML_Pipeline.arima import arima_model, tune_arima_order

# Import from lstm.py (new)
from src.ML_Pipeline.lstm import lstm_model

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
        r2, rmse, mape = regression(X_train, X_test, y_train, y_test, model_name)
        results[model_name] = {'R2': r2, 'RMSE': rmse, 'MAPE': mape}
        print(f"{model_name} - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
    
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
    
    arima_r2, arima_rmse, arima_mape, order, predictions, test = arima_model(df_global['avg_weekly_sales'].values, order=best_order)
    results['arima'] = {'R2': arima_r2, 'RMSE': arima_rmse, 'MAPE': arima_mape}
    print(f"ARIMA - R2: {arima_r2:.4f}, RMSE: {arima_rmse:.4f}, MAPE: {arima_mape:.4f}%")
    
    # Plot ARIMA predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('ARIMA Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Avg Weekly Sales')
    plt.legend()
    plt.savefig('artifacts/arima_predictions.png')
    plt.show()
    
    # For LSTM on the same global weekly mean 
    lstm_r2, lstm_rmse, lstm_mape, y_test_lstm, y_pred_lstm = lstm_model(df_global['avg_weekly_sales'].values)
    results['lstm'] = {'R2': lstm_r2, 'RMSE': lstm_rmse, 'MAPE': lstm_mape}
    print(f"LSTM - R2: {lstm_r2:.4f}, RMSE: {lstm_rmse:.4f}, MAPE: {lstm_mape:.4f}%")
    
    # Plot LSTM predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_lstm, label='Actual')
    plt.plot(y_pred_lstm, label='Predicted')
    plt.title('LSTM Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Avg Weekly Sales')
    plt.legend()
    plt.savefig('artifacts/lstm_predictions.png')
    plt.show()

    # Compare MAPE to find best model (or use RMSE as before)
    mape_dict = {model: metrics['MAPE'] for model, metrics in results.items()}
    best_model = min(mape_dict, key=mape_dict.get)
    print('Best model is {} having MAPE of {}%'.format(best_model, min(mape_dict.values())))

    # Add bar plots for R2, RMSE, MAPE
    models = list(results.keys())
    
    # R2 Bar Plot
    r2_values = [metrics['R2'] for metrics in results.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(models, r2_values, color='green')
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('Model Comparison based on R2 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('artifacts/model_r2_comparison.png')
    plt.show()
    
    # RMSE Bar Plot
    rmse_values = [metrics['RMSE'] for metrics in results.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(models, rmse_values, color='orange')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Model Comparison based on RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('artifacts/model_rmse_comparison.png')
    plt.show()
    
    # MAPE Bar Plot
    mape_values = [metrics['MAPE'] for metrics in results.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(models, mape_values, color='blue')
    plt.xlabel('Models')
    plt.ylabel('MAPE (%)')
    plt.title('Model Comparison based on MAPE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('artifacts/model_mape_comparison.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    run_pipeline()