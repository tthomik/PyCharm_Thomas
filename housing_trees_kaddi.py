import copy
import pandas as pd
import evaluation
import classifier.prune as cp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# PyCharm zeigt das komplette DataFrame an
pd.set_option('display.max_columns', 30)


# Einlesen der Daten
data = pd.read_csv("./data/Housing.csv")


# Transformierung der Daten
cleaned_data = data.dropna()
cleaned_data.reset_index(drop=True, inplace=True)

cat_encoded, categories = cleaned_data["ocean_proximity"].factorize()

input_data = cleaned_data.drop(["ocean_proximity"], axis=1)
input_data["ocean_proximity"] = cat_encoded

X = input_data.drop(["median_house_value"], axis=1)
y = input_data["median_house_value"]


# Splitten der Daten in Test, Validate und Train
X_data, test_data, X_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)

train_data, val_data, train_target, val_target = train_test_split(X_data, X_target, test_size=0.2, random_state=42)


# testen, welche depth den optimalen Baum liefert
numbers = range(1,26)
depth = []
R2_train = []

for number in numbers:
    depth.append(number)
    tree = DecisionTreeRegressor(max_depth=number)
    tree.fit(train_data, train_target)
    predict = tree.predict(val_data)
    predict_train = tree.predict(train_data)
    evaluation.append_errors(val_target, predict, name=number)
    r2_train = r2_score(train_target, predict_train)
    R2_train.append(r2_train)

max_depth_results = pd.DataFrame()
max_depth_results["Depth"] = depth
max_depth_results["MSE"] = evaluation.MSE
max_depth_results["RMSE"] = evaluation.RMSE
max_depth_results["R2-Score"] = evaluation.R2
max_depth_results["R2-Train"] = R2_train
max_depth_results["RMSE % of mean"] = evaluation.RMSE_of_mean
max_depth_results["Calibration"] = evaluation.Calibration

print(max_depth_results.sort_values(["R2-Score"], ascending=False).head())# max_depth = 8 ergibt den besten Baum


# Erzeuge neuen Baum, für's Prunen
tree10 = DecisionTreeRegressor(max_depth=10).fit(train_data,train_target)
tree10_data = tree10.tree_ # greife auf die Daten im Object "Tree" zu
lc = tree10_data.children_left # greife auf die linken Kinder zu (-1 ist ein Leaf)
rc = tree10_data.children_right # greife auf die rechten Kinder zu  (-1 ist ein Leaf)
is_node = (lc == rc)


# den oben erzeugten Baum prunen
tree_array = [tree10]
num_nodes = tree10.tree_.capacity
index = 0
alpha = 0
k = 1

while num_nodes > 1:
    tree_array.append(copy.deepcopy(tree_array[k - 1])) # jeder geprunte Baum wird als Object in den tree_array gespeichert
    min_node_idx, min_gk = cp.determine_alpha(tree_array[k].tree_)
    cp.prune(tree_array[k].tree_, min_node_idx)

    num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))
    k += 1


# wir predicten für denen geprunten Tree im tree_array auf die Validierungsdaten und berechnen den R2-Score
R2_score = []

for tree in range(0,len(tree_array)):
    tree_predict = tree_array[tree].predict(val_data)
    score = tree_array[tree].score(val_data, val_target)
    R2_score.append(score)

pruned_tree_results = pd.DataFrame()
pruned_tree_results["R2-Score"] = R2_score

print("")
print(pruned_tree_results.sort_values(["R2-Score"], ascending=False).head()) # Tree 566 ist der beste


# Berechnung des R2-Scores für ungeprunten Baum
R2_results = {}

predict_unpruned = tree10.predict(val_data)
R2_score_unpruned = tree10.score(val_data, val_target)


# Berechnung des Random Forests
random_forest = RandomForestRegressor(max_depth=9, random_state=42).fit(train_data, train_target)
R2_score_random_forest = random_forest.score(val_data, val_target)


# Zusammenfassung der R2-Scores
R2_results["ungepruned optimale max_depth"] = evaluation.R2[7]
R2_results["ungepruned"] = R2_score_unpruned
R2_results["pruned"] = R2_score[566]
R2_results["Random Forest"] = R2_score_random_forest

print("")
print(R2_results)

pd_results = pd.DataFrame.from_dict(R2_results)
print(pd_results)