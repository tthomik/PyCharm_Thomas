import pandas as pd
import classifier
import sklearn.linear_model as lm
import evaluation
import transform.manipulate as tm
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns",30)

if __name__ == '__main__':
    housing = pd.read_csv("data/housing.csv")

    # Entfernen der NaN
    housing = housing.dropna()

    # Reseten des index
    housing.reset_index(drop=True, inplace=True)

    # Splitten der Daten in Train und Test
    train, test = tm.train_test_split(housing,test_size=0.2, random_state=42)
    print (train.head())

    #Factorizen von "ocean_proximity"
    train_factorized, train_categories = tm.factorize(train,"ocean_proximity")
    train_cat_pd = pd.DataFrame(train_factorized)
    train_cat_pd = train_cat_pd.add_prefix("ocean_proximity")



    x_train = train.drop("median_house_value", axis=1)
    y_train = train["median_house_value"]

    randomForest = RandomForestRegressor()
    randomForest.fit(x_train,y_train)
    y_train_predict = randomForest.predict(x_train, y_train)

