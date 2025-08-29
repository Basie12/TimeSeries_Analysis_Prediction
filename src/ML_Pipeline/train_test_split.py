import pandas as pd 
from sklearn.model_selection import train_test_split

def data_split(X,y, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    return X_train, X_test, y_train, y_test