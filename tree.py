import pandas as pd
import seaborn as sns
import copy

from sklearn.tree import DecisionTreeRegressor, tree, export_graphviz

from classifier.prune import determine_alpha, prune

sns.set()

if __name__ == '__main__':
    data = pd.read_csv("data/housing.csv")

    input_features = data[['latitude', 'longitude']]
    target = data['median_house_value'] / 100000

    d_tree = DecisionTreeRegressor(max_depth=2)
    d_tree.fit(input_features, target)

    # Objekt d_tree (gesamter Baum) wird in dem tree_array gespeichert
    tree_array = [d_tree]

    # .tree = tree object ->  aufrufen der Tree-Klasse, capacity=sowas wie Anzahl der Knoten
    num_nodes = d_tree.tree_.capacity
    index = 0
    alpha = 0
    k = 1

    # While-Schleife: solange num_nodes >1, mache:
    while num_nodes > 1:
        #deepcopy = eine Kopie, die unabhängig vom Original ist
        tree_array.append(copy.deepcopy(tree_array[k - 1]))

        min_node_idx, min_gk = determine_alpha(tree_array[k].tree_)

        prune(tree_array[k].tree_, min_node_idx)

        num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))

        k += 1


    if False:
        for k in range(0,len(tree_array)):
            export_graphviz(tree_array[k], out_file='tree' + str(k) + '.dot')

