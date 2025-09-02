# TimeSeries_Analysis_Prediction

## Project Description

This project is a time series analysis and prediction pipeline for forecasting average weekly sales in an FMCG (Fast-Moving Consumer Goods) dataset. It processes daily sales data, aggregates it to weekly levels, performs feature engineering (including lag features, rolling means, seasonal encoding, and hierarchy averages), handles encoding and scaling, and evaluates multiple machine learning and time series models to identify the best performer based on metrics like R2, MAE, and MSE.

Key features:
- Data aggregation from daily to weekly per SKU.
- Feature engineering: Lags, rolling averages, seasonal indicators, one-hot encoding for categorical features, standard scaling, and correlation-based feature selection.
- Models evaluated: Linear Regression, Random Forest, XGBoost, ARIMA, and LSTM (deep learning for sequence modeling).
- Pipeline integration: A single `engine.py` script runs data processing, feature engineering, model training, evaluation, and comparison.
- Artifacts: Saves preprocessors, scalers, models, and processed data for reproducibility and inference.

The project is structured under `src/ML_Pipeline/` with modular scripts for each component, making it easy to extend or modify.

## Installation

### Python Version
This project requires Python 3.10 for compatibility with all dependencies.

### Creating a Virtual Environment and Installing Requirements

#### For Windows:
1. Open Command Prompt (Win + R, type "cmd", Enter).
2. Navigate to your project directory:
3. Create a virtual environment:
    * python -m venv ts_env
4. Activate it:
    * ts_env\Scripts\activate
5. Install requirements:
    * pip install -r requirements.txt


### For Linux/Mac:
O#### For Linux/Mac:
1. Open a terminal.
2. Navigate to your project directory: cd /path/to/TimeSeries_Analysis_Prediction
3. Create a virtual environment: python3.10 -m venv ts_env
4. Activate it: source ts_env/bin/activate
5. Install requirements:pip install -r requirements.txt


ariable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions, use `py -3.10` (Windows) or `python3.10` (Linux/Mac) to specify the version.

Note: Ensure Python 3.10 is installed and in your PATH. For TensorFlow on Mac (Intel), this setup works; for Apple Silicon, consider additional steps like `tensorflow-metal`.

## Usage

### Running the Pipeline
1. Activate your virtual environment (as above).
2. From the project root, run:



- This executes the full pipeline: data loading, aggregation, feature engineering, model training/evaluation, and comparison.
- Outputs: Model metrics (R2, MAE, MSE), best model identification, saved artifacts (e.g., models in `artifacts/`, processed data).

### Configurable Elements
- Edit `src/ML_Pipeline/config.py` for file paths (e.g., `DATA_FILEPATH`, `PROCESSED_DATA_FILEPATH`).
- For custom runs, modify `engine.py` (e.g., uncomment ARIMA tuning, adjust LSTM hyperparameters like `n_steps` or epochs).

### Example Output
The script will print model performances and the best model, e.g.:



## Project Structure
TimeSeries_Analysis_Prediction/
├── src/
│   ├── engine.py                 # Main pipeline runner
│   └── ML_Pipeline/
│       ├── config.py             # Configuration (paths, column names)
│       ├── dataset.py            # Data loading and aggregation
│       ├── Featureengineering.py # Feature creation, encoding, scaling
│       ├── regression_models.py  # ML models (Linear, RF, XGBoost)
│       ├── arima.py              # ARIMA model implementation
│       ├── lstm.py               # LSTM model implementation
│       └── ...                   # Other utils if added
├── artifacts/                    # Saved models, preprocessors
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── ...                           # Data files, outputs

## Models Used
- **Machine Learning**: Linear Regression, Random Forest, XGBoost (feature-based regression).
- **Time Series**: ARIMA (autoregressive integrated moving average for univariate forecasting).
- **Deep Learning**: LSTM (long short-term memory network for sequential data).

Models are evaluated on a test split, with ARIMA and LSTM applied to global weekly averages for simplicity.

## Visuals Recommendation
Yes, including visuals in your README or project would enhance it, especially for a time series analysis project. Here's what I suggest:

1. **Correlation Heatmap**: From feature engineering (already in code via seaborn.heatmap). Add a screenshot showing feature correlations to highlight multicollinearity removal.
2. **Sales Trends Plot**: Plot weekly sales over time (e.g., using matplotlib or seaborn in a notebook). This visualizes seasonality or trends in the data.
3. **Model Performance Comparison**: Bar chart of MSE/R2 across models (generate via matplotlib.bar in engine.py or a separate script).
4. **Forecast vs Actual Plot**: For ARIMA/LSTM, plot predictions vs test data to show forecasting accuracy.
5. **Feature Importance**: For XGBoost/RF, plot top features (using model.feature_importances_).

To include:
- Run code to generate images (e.g., save with plt.savefig('visuals/corr_heatmap.png')).
- Add to README: `![Correlation Heatmap](visuals/corr_heatmap.png)`
- Create a `visuals/` folder and commit the images (or generate them dynamically in a Jupyter notebook for demo).

This makes the project more engaging and demonstrates results visually.

## Contributing
Feel free to fork and submit pull requests. Ensure tests pass and follow Python PEP8 style.

## License
MIT License (or specify your own).