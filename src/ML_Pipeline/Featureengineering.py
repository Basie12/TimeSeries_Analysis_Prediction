import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DATA_FILEPATH, DATE_COL_NAME, PROCESSED_DATA_FILEPATH

# Import functions from dataset.py (assuming it's in the same directory)
from .dataset import read_data, aggregate_to_weekly, compute_hierarchy_averages, merge_hierarchy_averages

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def feature_engineering(input_filepath, date_col_name, output_filepath):
    # Load and process data using functions from dataset.py
    df_daily = read_data(input_filepath, date_col_name)
    df_weekly = aggregate_to_weekly(df_daily)
    category_avg, segment_avg, brand_avg = compute_hierarchy_averages(df_daily)
    df_processed = merge_hierarchy_averages(df_weekly, category_avg, segment_avg, brand_avg)
    
    # Add season feature
    df_processed['season'] = df_processed['date'].dt.month.apply(get_season)
    
    # Optionally add other date features (user code references 'month' and 'year' in removal, so add them)
    df_processed['year'] = df_processed['date'].dt.year
    df_processed['month'] = df_processed['date'].dt.month
    
    # Sort by sku and date
    df_processed = df_processed.sort_values(['sku', 'date'])
    
    # Add lag and rolling features (not in dataset.py, so include)
    for lag in [1, 4]:
        df_processed[f'lag_{lag}'] = df_processed.groupby('sku')['avg_weekly_sales'].shift(lag)
    df_processed['rolling_mean_4'] = df_processed.groupby('sku')['avg_weekly_sales'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    
    # Drop NaNs from lags
    df_processed = df_processed.dropna()
    
    # Remove unnecessary columns
    cols_to_remove = ['date', 'sku', 'month', 'year']  # As per user code
    df_processed = df_processed.drop(columns=cols_to_remove, errors='ignore')
    
    # Define categorical and numerical before encoding for correlation analysis post-encoding
    categorical_cols = ['brand', 'segment', 'category', 'channel', 'region', 'pack_type', 'season']
    numerical_cols = [col for col in df_processed.columns if col not in categorical_cols and col != 'avg_weekly_sales']
    
    # Create preprocessor for encoding (use OneHotEncoder for consistency and saving)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'  # Keep numerical as is for now
    )
    
    # Prepare X and y
    y = df_processed['avg_weekly_sales']
    X = df_processed.drop(columns=['avg_weekly_sales'])
    
    # Transform to get encoded data for correlation analysis
    X_transformed = preprocessor.fit_transform(X)
    encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_cols = list(encoded_cat_cols) + numerical_cols  # Order: encoded first, then numerical
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_cols, index=X.index)
    
    # Compute correlation matrix on transformed data (all numeric now)
    corr = X_transformed_df.corr()
    print(corr)
    
    # Plot the correlation matrix
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    plt.show()
    
    # Get features
    features = list(X_transformed_df.columns)
    print(features)
    
    # Remove based on high correlation (>0.8)
    corr_abs = corr.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"Columns to remove due to high correlation: {to_drop_corr}")
    
    # Drop from transformed_df
    keep_cols = [col for col in features if col not in to_drop_corr]
    X_selected = X_transformed_df[keep_cols]
    
    # Now apply standard scaling to the selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=keep_cols, index=X_selected.index)
    
    # Combine with target (sku and date were dropped earlier, but if needed for model.py, can re-add from original index or save separately)
    df_final = pd.concat([X_scaled_df, y], axis=1)
    
    # Save the processed data
    df_final.to_csv(output_filepath, index=False)
    
    # Save the preprocessor and scaler for potential inference in model.py
    joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    # Save keep_cols for feature selection in inference
    joblib.dump(keep_cols, 'artifacts/keep_cols.pkl')
    
    return df_final

# Example execution (optional for testing)
if __name__ == "__main__":
    from config import DATA_FILEPATH, DATE_COL_NAME  # Reuse from dataset.py
    # Define output path in config or hardcode for testing
    #  PROCESSED_DATA_FILEPATH = 'processed_data.csv' # Example; add to config.py
    df_engineered = feature_engineering(DATA_FILEPATH, DATE_COL_NAME, PROCESSED_DATA_FILEPATH)
    print(df_engineered.head())