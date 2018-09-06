import pandas as pd
import classifier
import sklearn.linear_model as lm
import evaluation
import transform.manipulate as tm
pd.set_option("display.max_columns",30)

if __name__ == '__main__':
    heart = pd.read_csv("data/heart.csv")
    print(heart.isnull().sum())

    #Entfernen der NaN
    heart = heart.dropna()
    #Reseten des index
    heart.reset_index(drop=True, inplace=True)

    # Umwandlung von "Yes" in 1 und "No" in 0
    heart["AHD"] = 1.0 * (heart["AHD"] == "Yes")

    # Aufteilung des Datensatzes in Train und Test
    train, test = tm.split_train_test(heart,0.2,42, "AHD")

    # Datensatz nur mit numerischen Merkmalen
    train_num = train.drop(labels = ["ChestPain", "Thal", "Sex", "Fbs", "RestECG", "ExAng", "Slope", "Unnamed: 0", "AHD"], axis=1)
    test_num = test.drop(labels = ["ChestPain", "Thal", "Sex", "Fbs", "RestECG", "ExAng", "Slope", "Unnamed: 0", "AHD"], axis=1)

    # Datensatz nur mit kategorischen Merkmalen
    train_cat = train.drop(labels = ["Age","RestBP","Chol","MaxHR","Oldpeak","Ca", "Unnamed: 0"], axis=1)
    test_cat = test.drop(labels=["Age", "RestBP", "Chol", "MaxHR", "Oldpeak", "Ca", "Unnamed: 0"], axis=1)#


    # Skalieren des numerischen Datensatzes mit StandardScaler
    train_num_scaled = tm.std_scaler(train_num)
    test_num_scaled = tm.std_scaler(test_num)

    # Factorizen der kategorischen Merkmale
    train_chestpain_factorized = tm.factorize(train_cat,"ChestPain")
    train_thal_factorized = tm.factorize(train_cat, "Thal")
    train_chestpain_factorized = tm.factorize(train_cat, "ChestPain")
    train_thal_factorized = tm.factorize(train_cat, "Thal")

    # OneHotEncoden der kategorischen Merkmale
    #train_chestpain_1hot = tm.onehotencode(train_cat,"ChestPain")
    #print(train_chestpain_1hot)