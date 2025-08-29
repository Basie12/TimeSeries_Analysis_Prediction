# e.g., in models.py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from config import MODEL_BASE_PATH 

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
    
    pipe = Pipeline([('scaler', StandardScaler()), (model_name, model_dict[model_name]())])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe.named_steps[model_name], model_path)
    print(f'model saved in {model_path}')
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse