import numpy as np
import transform.manipulate as tm
import pandas as pd
pd.set_option("display.max_columns",30)
import json

if __name__ == '__main__':


    file = open("data/imdb/train_data.csv", "r")
    train_data = [list(map(int,line.split(','))) for line in file]

    file = open("data/imdb/test_data.csv", "r")
    test_data = [list(map(int,line.split(','))) for line in file]

    train_labels = pd.read_csv("data/imdb/train_labels.csv", header=None)
    test_labels = pd.read_csv("data/imdb/test_labels.csv", header=None)

    f_word_index = open('data/imdb/word_index.json', "r")
    word_index = json.loads(f_word_index.read())

    f_reverse_word_index = open('data/imdb/reverse_word_index.json', "r")
    reverse_word_index = json.loads(f_reverse_word_index.read())

    reverse_word_index = {int(key):reverse_word_index[key] for key in reverse_word_index}

    # Indizes 0, 1, 2 werden für padding", "start of sequence", und "unknown" verwendet, deshalb verschieben wir den Index bei Dekodieren um 3:
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
    decoded_review

    # One Hot Encoding
    """Wir wissen, dass unser Wörterbuch nur 10,000 Worte umfasst und können daher jede Sequenz als einen 
    10,000 dimensionalen Vektor von 0 und 1 kodieren. Dabei bedeutet jede Position in diesem Vektor, 
    dass das Wort vorgekommen ist."""

    def vectorize_sequences(sequences, dimension=10000):
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # set specific indices of results[i] to 1s
        return results

    # Our vectorized training data
    print(type(train_data))
    #x_train = vectorize_sequences(train_data)
    print(train_data[])

    # Our vectorized test data
    #x_test = vectorize_sequences(test_data)

    # Our vectorized labels
    #y_train = np.asarray(train_labels).astype('float32')
    #y_test = np.asarray(test_labels).astype('float32')