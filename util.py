import numpy as np
import pandas as pd

# data information check
def data_info(df):
    '''
    check the size of data: rows, columns
    check the missing value info
    '''
    print('----------Information of the data----------')
    print('Number of Rows:', len(df), '  ', 'Number of Columns:', len(df.columns))
    print('Column names:')
    print(df.columns)
    print('Missing value information:')
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            print('{}: '.format(i), 'the number of missing values is', 
                  df[i].isnull().sum())
    

class Dummy_Transformer(object):
    
    def __init__(self, column_name):
        self.keys = None
        self.column_name = column_name
    
    def fit(self, X, y=None):
        self.keys = set(X)
    
    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[self.column_name + key] = [0]*len(X)    
        for i, item in enumerate(X):
            if item in self.keys:
                res[self.column_name + item][i] = 1
        return pd.DataFrame(res)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

