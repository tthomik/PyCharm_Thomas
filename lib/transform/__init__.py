import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def add(a, b):
    return a + b


def sum_list(the_list):
    return np.sum(the_list)


def say_hello():
    print('hello')


def say_no():
    print('no')

# Factorizen eines kategorischen Attributs
def factorize(df, Spalte):
    df_cat = df[Spalte]
    df_cat_encoded, df_categories = df_cat.factorize()
    return df_cat_encoded, df_categories

# OneHotEncoden eines kategorischen Merkmals, das zuvor gefactorized wurde
# Umwandlung von Sparse Matrix zu numpy array
# Umwandlung von numpy array zu Pandas DataFrame
def onehotencode(df_cat_encoded, categories):
    encoder = OneHotEncoder()
    df_1hot = encoder.fit_transform(df_cat_encoded.reshape(-1, 1))
    df_1hot_np = df_1hot.toarray()
    df_1hot_pd = pd.DataFrame(df_1hot_np)
    df_1hot_pd.columns=categories
    return df_1hot_pd
