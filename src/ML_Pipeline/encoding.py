import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def label_encoding(dataset, columns_list):
    le = LabelEncoder()
    for col in columns_list:
        dataset[col] = le.fit_transform(dataset[col])
    return dataset

def one_hot_encoding(dataset, columns_list):
    ohe = pd.get_dummies(data=dataset, columns=['region'])
    return ohe



# def train_test(dataset):
#     # Encoding labels for 'type'
#     le = LabelEncoder()
#     dataset['type'] = le.fit_transform(dataset['type'])

#     # One Hot Encoding for region
#     ohe = pd.get_dummies(data=dataset, columns=['region'])

#     X = ohe.drop(['AveragePrice', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
#     y = dataset['AveragePrice']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#     return X_train, X_test, y_train, y_test
