# prophet_model.py - Example script incorporating config.py globals

import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import DATA_FILEPATH, MODEL_BASE_PATH, DATE_COL_NAME  # Import globals from config.py

def preprocess_data(filepath=DATA_FILEPATH, date_col_name=DATE_COL_NAME):
    # Load daily data
    daily_df = pd.read_csv(filepath)
    daily_df[date_col_name] = pd.to_datetime(daily_df[date_col_name])
    daily_df['year'] = daily_df[date_col_name].dt.year
    daily_df = daily_df[daily_df['year'] != 2025]  # Filter out 2025
    daily_df.info()
    
    # Aggregate to weekly level per SKU
    df_weekly = daily_df.set_index(date_col_name).groupby(['sku', pd.Grouper(freq='W')]).agg({
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
    
    # Rename target
    df_weekly.rename(columns={'units_sold': 'avg_weekly_sales'}, inplace=True)
    
    # Create global weekly average for Prophet (mean across SKUs)
    df_global_weekly = df_weekly.groupby(date_col_name)[['avg_weekly_sales']].mean().reset_index()
    df_global_weekly.columns = ['ds', 'y']  # Prophet format
    
    return df_weekly, df_global_weekly

def prophet_pipeline(prediction_size=50, regressor_cols=None, holidays=None, model_path=f"{MODEL_BASE_PATH}/prophet_model.pkl"):
    # Preprocess data using globals
    _, df_global_weekly = preprocess_data()
    
    # Train/test split for evaluation
    train_size = len(df_global_weekly) - prediction_size
    df_train = df_global_weekly.iloc[:train_size].copy()
    df_test = df_global_weekly.iloc[train_size:].copy()

    # Initialize Prophet
    m = Prophet(holidays=holidays, weekly_seasonality=True, daily_seasonality=False)

    # Add regressors if provided (aggregated to global weekly mean)
    if regressor_cols:
        for col in regressor_cols:
            m.add_regressor(col)
            # Aggregate regressors to global weekly
            df_global_weekly[col] = df_weekly.groupby(date_col_name)[col].mean()

    # Fit on train
    m.fit(df_train)

    # Save model using global path
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(m, f)
    print(f"Model saved to {model_path}")

    # Future DataFrame for forecast
    future = m.make_future_dataframe(periods=prediction_size, freq='W')
    if regressor_cols:
        for col in regressor_cols:
            future[col] = df_global_weekly[col].mean()  # Use mean for future; replace with planned values

    # Forecast
    forecast = m.predict(future)

    # Evaluate on test if prediction_size allows
    if prediction_size > 0:
        forecast_test = forecast.iloc[-prediction_size:]['yhat']
        y_true = df_test['y']
        r2 = r2_score(y_true, forecast_test)
        mae = mean_absolute_error(y_true, forecast_test)
        mse = mean_squared_error(y_true, forecast_test)
        print(f"R2: {r2:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}")
    else:
        print("No test set for evaluation.")
    
    # Plot
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    
    return m, forecast, df_global_weekly

# Example call
if __name__ == "__main__":
    # Optional holidays (US example)
    holidays = pd.DataFrame({
        'holiday': 'us_holiday',
        'ds': pd.to_datetime([
            '2022-01-01', '2022-07-04', '2022-11-24', '2022-11-25', '2022-12-25',
            '2023-01-01', '2023-07-04', '2023-11-23', '2023-11-24', '2023-12-25',
            '2024-01-01', '2024-07-04', '2024-11-28', '2024-11-29', '2024-12-25'
        ]),
        'lower_window': 0,
        'upper_window': 1
    })
    
    m, forecast, df_global_weekly = prophet_pipeline(prediction_size=50, holidays=holidays)