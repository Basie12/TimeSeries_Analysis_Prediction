from ML_Pipeline.dataset import read_data
from ML_Pipeline.encoding import label_encoding, one_hot_encoding
from ML_Pipeline.train_test_split import train_test_split 
from ML_Pipeline.regression_models import regression
from ML_Pipeline.get_best_arima_params import best_d_value, evaluate_models
from ML_Pipeline.arima import arima_model
# from ML_Pipeline.prophet import prophet

df = read_data(filepath='../input/avocado.csv', date_col_name='Date')
df = label_encoding(dataset=df, columns_list=['type'])
df = one_hot_encoding(dataset=df, columns_list=['region'])

X = df.drop(['AveragePrice', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
y = df['AveragePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('Preprocess done!')

lr_score = regression(X_train, X_test, y_train, y_test, 'lin_reg', '../output/lin_reg.pkl')
rf_score = regression(X_train, X_test, y_train, y_test, 'rf_reg', '../output/rf_reg.pkl')
xgb_score = regression(X_train, X_test, y_train, y_test, 'xgb_reg', '../output/xgb_reg.pkl')

# arima
data = read_data(filepath='../input/avocado.csv', date_col_name='Date')

df_ar = data.drop(columns=['Total Volume', '4046', '4225', '4770', 'Total Bags',
                            'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region'])
df_ar = df_ar.resample('W').mean()
best_d_value(df_ar['AveragePrice'])
# evaluate parameters
p_values = range(1, 2)
d_values = range(0, 4)
q_values = range(0, 2)
# best_order = evaluate_models(df_ar.values, p_values, d_values, q_values)
best_order = (1,0,0)
arima_score = arima_model(df_ar,best_order )


# Prophet
# df_pr = data[['AveragePrice']]
# df_pr = df_pr.resample('W').mean()
# df_pr.reset_index(inplace=True)  # Removing the datetime index
# df_pr = df_pr.rename(columns={'Date': 'ds', 'AveragePrice': 'y'})
# prophet_score = prophet(data=df_pr, y_col_name='y', date_col_name='ds', pred_col_name='yhat')
# print('Prophet done')

mse_dict = {'Linear Regression': lr_score[2],
            'Random Forest': rf_score[2],
            'XGBoost': xgb_score[2],
            'ARIMA': arima_score[2],
            }

print('Best model is {} having MSE of {}'.format(min(mse_dict, key=mse_dict.get),
                                                    min(mse_dict.values())))