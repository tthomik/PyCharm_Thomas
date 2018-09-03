#python main.py

# 1. Einen Ordner anlegen
# 2. housing.csv in den Ordner kopieren
# 3. Eine Funktion zum Einlesen der Datei erstellen

import utils


if __name__ =="__main__":
    heart = utils.read_csv("data/Heart.csv")
    print(heart.head())
    print(type(heart))




