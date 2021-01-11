import numpy as np
import random
from keras.utils.np_utils import to_categorical


def k_validation_split(k, dataset):
    data = np.copy(dataset)
    folds = []
    fold_size = data.shape[0] // k
    for i in range(k):
        fold = []
        x = 1 if data.shape[0] % fold_size > 0 else 0
        for j in range(fold_size + x):
            pop_index = random.randrange(data.shape[0])
            fold.append(data[pop_index].tolist())
            data = np.delete(data, pop_index, 0)
        fold = np.array(fold)
        folds.append(fold)
    return folds


def split_output(data):
    X = data[:, :-1]
    y = to_categorical(data[:, -1])
    return X, y


def calculate_performance(test_y, predicted_class):
    return (np.array(predicted_class) == np.array(test_y)).mean()
