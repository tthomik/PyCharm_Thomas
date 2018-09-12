import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Factorizen von kategorischen Merkmalen in Zahlen
def factorize(df, column):
    df = df[column]
    df_factorized, df_categories = df.factorize()
    return df_factorized, df_categories

# OneHotEncoden von gefactorizten Merkmalen oder kategorischen Merkmalen, die als Zahlen angegeben sind
def onehotencode(df, column):
    encoder = OneHotEncoder()
    df = df[column]
    df = encoder.fit_transform(df[column].value.reshape(-1,1))
    df_np = df.toarray()
    df_pd = pd.DataFrame(df_np)
    df_pd = df_pd.add_prefix(column)
    return df_pd

# Factorizen und OneHotEncoden in einem Schritt
def umwandlung_binaer(df):
    # Schaut in DataFrames, welche Spalten Objekte enhalten und speichert sie in objects
    objects = df.select_dtypes(include=[object])

    # Speichert Anzahl der Spalten in sum_columns
    sum_columns = len(objects.columns)

    # Values aus den Columns werden genommen und zu objects_list hinzugefügt
    objects_list = objects.columns.values.tolist()

    df_dropped = df.drop(labels=objects_list, axis=1, errors='ignore')

    for column in range(0, (sum_columns)):
        cat = objects.iloc[:, column]
        cat_encoded, categories = cat.factorize()
        list_categories = categories.tolist()
        encoder = OneHotEncoder()
        cat_1hot = encoder.fit_transform(cat_encoded.reshape(-1, 1))
        np_cat_1hot = cat_1hot.toarray()
        pd_cat_1hot = pd.DataFrame(np_cat_1hot)

        # Columns von pd_cat_1hot sollen heißen wie Objekte in list_categories
        pd_cat_1hot.columns = list_categories

        df_dropped.reset_index(drop=True, inplace=True)
        df = pd.concat((df_dropped, pd_cat_1hot), axis=1)

    return df


# Split in Train und Test mit stratify, wenn label kategorisch ist
def split_train_test(df,size,state,column):
    train, test = train_test_split(df, test_size=size,random_state=state)
    train = train.sort_index()
    test = test.sort_index()
    return train, test

# Standardscaler (skalieren von numerischen Attributen)
def std_scaler(df):
    scaler = StandardScaler(copy=True, with_mean=True, with_std = True)
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled_pd = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled_pd

# Min-Max-Skalierer
def min_max_scaler(df,range):
    scaler = min_max_scaler(feature_range=(range))
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled_pd = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled_pd
