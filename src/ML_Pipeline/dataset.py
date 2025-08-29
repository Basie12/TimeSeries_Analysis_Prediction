import pandas as pd
import numpy as np
from config import DATA_FILEPATH, DATE_COL_NAME  

def read_data(filepath, date_col_name):
    try:
        # Read CSV without index_col=0 (assume 'date' is a column)
        dataset = pd.read_csv(filepath)
        
        # Convert to datetime
        dataset[date_col_name] = pd.to_datetime(dataset[date_col_name], errors='coerce')
        
        # Drop invalid dates
        dataset = dataset.dropna(subset=[date_col_name])
        
        # Set index
        dataset.set_index(date_col_name, inplace=True)
        
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    except KeyError:
        raise KeyError(f"The column '{date_col_name}' does not exist in the dataset.")
    except ValueError as ve:
        raise ValueError(f"Error parsing dates: {ve}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def aggregate_to_weekly(df_daily):
    df_weekly = df_daily.groupby(['sku', pd.Grouper(key='date', freq='W')]).agg({
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
    
    return df_weekly

def compute_hierarchy_averages(df_daily):
    category_avg = df_daily.groupby(['category', pd.Grouper(key='date', freq='W')])['units_sold'].mean().reset_index()
    category_avg.rename(columns={'units_sold': 'category_avg_sales'}, inplace=True)
    
    segment_avg = df_daily.groupby(['segment', pd.Grouper(key='date', freq='W')])['units_sold'].mean().reset_index()
    segment_avg.rename(columns={'units_sold': 'segment_avg_sales'}, inplace=True)
    
    brand_avg = df_daily.groupby(['brand', pd.Grouper(key='date', freq='W')])['units_sold'].mean().reset_index()
    brand_avg.rename(columns={'units_sold': 'brand_avg_sales'}, inplace=True)
    
    return category_avg, segment_avg, brand_avg

def merge_hierarchy_averages(df_weekly, category_avg, segment_avg, brand_avg):
    df_weekly = df_weekly.merge(category_avg, on=['date', 'category'], how='left').fillna(0)
    df_weekly = df_weekly.merge(segment_avg, on=['date', 'segment'], how='left').fillna(0)
    df_weekly = df_weekly.merge(brand_avg, on=['date', 'brand'], how='left').fillna(0)
    return df_weekly

# Example execution (optional for testing)
if __name__ == "__main__":
    df_daily = read_data(DATA_FILEPATH, DATE_COL_NAME)
    df_weekly = aggregate_to_weekly(df_daily)
    category_avg, segment_avg, brand_avg = compute_hierarchy_averages(df_daily)
    df_weekly = merge_hierarchy_averages(df_weekly, category_avg, segment_avg, brand_avg)
    print(df_weekly.head())