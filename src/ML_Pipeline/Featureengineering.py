# regression_models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
from config import MODEL_BASE_PATH 

# Define models
model_dict = {
    'lin_reg': LinearRegression,
    'rf_reg': RandomForestRegressor,
    'xgb_reg': XGBRegressor
}

def regression_cv(X, y, model_name, n_splits=5, model_path=None, plot_importance=True):
    """
    Train and evaluate regression model using TimeSeriesSplit CV.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model_name (str): One of ['lin_reg', 'rf_reg', 'xgb_reg']
        n_splits (int): Number of CV splits
        model_path (str): Optional save path, defaults to MODEL_BASE_PATH
        plot_importance (bool): If True, plot feature importance for tree models

    Returns:
        dict: avg metrics and best model pipeline
    """
    if model_name not in model_dict:
        raise ValueError(f'Only these options for model_name are allowed: {list(model_dict.keys())}')

    if model_path is None:
        model_path = f"{MODEL_BASE_PATH}{model_name}_model.pkl"

    # Conditional scaler: needed for LinearRegression, not for tree models
    steps = []
    if model_name == 'lin_reg':
        steps.append(('scaler', StandardScaler()))
    steps.append((model_name, model_dict[model_name]()))
    pipe = Pipeline(steps)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {"r2": [], "mae": [], "mse": []}
    feature_importances = []

    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics["r2"].append(r2_score(y_test, y_pred))
        metrics["mae"].append(mean_absolute_error(y_test, y_pred))
        metrics["mse"].append(mean_squared_error(y_test, y_pred))

        # Collect feature importances if tree model
        if model_name in ['rf_reg', 'xgb_reg']:
            importances = pipe.named_steps[model_name].feature_importances_
            feature_importances.append(importances)

        print(f"Fold {fold} → R2={metrics['r2'][-1]:.3f}, MAE={metrics['mae'][-1]:.3f}, MSE={metrics['mse'][-1]:.3f}")
        fold += 1

    # Aggregate metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\nAverage Performance ({model_name}): {avg_metrics}")

    # Train final model on full dataset
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    print(f"Final model saved to {model_path}")

    # Aggregate feature importances across folds
    if feature_importances:
        mean_importances = np.mean(feature_importances, axis=0)
        feat_imp_series = pd.Series(mean_importances, index=X.columns).sort_values(ascending=False)

        if plot_importance:
            plt.figure(figsize=(10, 5))
            feat_imp_series.plot(kind='bar')
            plt.title(f"{model_name} - Aggregated Feature Importances")
            plt.ylabel("Mean Importance")
            plt.show()

        return {"metrics": avg_metrics, "pipeline": pipe, "feature_importances": feat_imp_series}

    return {"metrics": avg_metrics, "pipeline": pipe}
