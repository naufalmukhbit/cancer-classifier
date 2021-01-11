import pandas as pd
import numpy as np
import random
from copy import deepcopy

# BP Essentials
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

import pyswarms as ps

def class_to_float(data, class0, class1, class2):
  for i in range(data.shape[0]):
    if class2 == None:
      data.iloc[i, -1] = 1.0 if data.iloc[i, -1] == class1 else 0.0
    else:
      data.iloc[i, -1] = 1.0 if data.iloc[i, -1] == class1 else 2.0 if data.iloc[i, -1] == class2 else 0.0
    
def min_max_normalization(np_data):
  data_temp = np_data.transpose()
  minmax = []
  for i in range(data_temp.shape[0] - 1):
    minmax.append([data_temp[i].min(), data_temp[i].max()])
  for j in range(np_data.shape[0]):
    for k in range(np_data.shape[1] - 1):
      np_data[j][k] = 0 if (minmax[k][1] == minmax[k][0]) else (np_data[j][k] - minmax[k][0])/(minmax[k][1] - minmax[k][0])

lung_train = pd.read_csv('lungCancer_train.data', header = None)
lung_test = pd.read_csv('lungCancer_test.data', header = None)

# changing 'ADCA' to 1 and 'Mesothelioma' to 0
class_to_float(lung_train, 'Mesothelioma', 'ADCA', None)
class_to_float(lung_test, 'Mesothelioma', 'ADCA', None)

# convert dataframe to numpy array to normalize
lung_train_np = lung_train.to_numpy()
min_max_normalization(lung_train_np)

lung_test_np = lung_test.to_numpy()
min_max_normalization(lung_test_np)

def initialize_network(n_features, n_classes, neurons):
  # Initialize network size
  inputs = Input(shape=(n_features,))
  hidden_layer = None
  for i in neurons:
    hidden_layer = Dense(i, activation='sigmoid')(inputs if hidden_layer == None else hidden_layer)
  outputs = Dense(n_classes, activation='sigmoid')(hidden_layer)

  # Initialize network
  network = Model(inputs=inputs, outputs=outputs)
  network.compile(optimizer=SGD(lr=0.1), loss='mean_squared_error')

  return network

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
  X = data[:,:-1]
  y = to_categorical( data[:,-1] )

  return X, y

def update_network(network, pos):
  new_weights = [layer.get_weights() for layer in network.layers]
  new_weights[1][0] = new_weights[1][0][pos==1,:]

  n_classes = network.layers[-1].get_weights()[1].shape[0]
  hidden_layers = [i.get_weights()[1].shape[0] for i in network.layers[1:-1]]
  new_network = initialize_network(np.count_nonzero(pos), n_classes, hidden_layers)
  for i in range(len(new_weights)):
    new_network.layers[i].set_weights(new_weights[i])

  return new_network

def f(x, network, X, y, alpha=0.88):
  n_particles = x.shape[0]
  objective = []

  # Calculate objective function per-particle
  for i in range(n_particles):
    if np.count_nonzero(x[i]) == 0:
      X_subset = X
      # temp_network = deepcopy(network)
    else:
      X_subset = X*x[i]
      # temp_network = update_network(network, x[i])
    P = calculate_performance(y, predict_data(network, X_subset))
    objective.append((alpha * (1.0 - P)) + ((1.0 - alpha) * (X_subset.shape[1] / X.shape[1])))

  return np.array(objective)

def reduce_dimensionality(network, X, y, epochs, n_particles, options):
  optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=X.shape[1], options=options)

  # Perform optimization
  cost, pos = optimizer.optimize(f, network=network, X=X, y=y, iters=epochs)
  return pos

def predict_data(model, test_X):
  prediction = model.predict(test_X)
  return [np.argmax(i) for i in prediction]

def calculate_performance(test_y, predicted_class):
  actual_class = [np.argmax(i) for i in test_y]
  return (np.array(predicted_class) == np.array(actual_class)).mean()

def evaluate_algorithm(k, dataset, options):
  folds = k_validation_split(k, dataset)
  scores = []
  for i in range(len(folds)):
    print("Iteration ", i+1, " of ", len(folds), " folds.")
    trainset = deepcopy(folds)
    testset = trainset.pop(i)
    trainset = np.concatenate((trainset[0], trainset[1], trainset[2], trainset[3]))

    train_X, train_y = split_output(trainset)
    test_X, test_y = split_output(testset)

    network = initialize_network(train_X.shape[1], train_y.shape[1], [16])
    pos = reduce_dimensionality(network, train_X, train_y, epochs=50, n_particles=30, options=options)

    new_network = update_network(network, pos)
    train_X = train_X[:,pos==1]
    test_X = test_X[:,pos==1]

    new_network.fit(train_X, train_y, epochs=100, verbose=1)
    scores.append(calculate_performance(test_y, predict_data(new_network, test_X)))
  return scores


# Dataset for Lung Cancer
dataset = np.concatenate((lung_train_np, lung_test_np))

# c1: Cognitive Parameter
# c2: Social Parameter
# w: Inertia Parameter
# k: Number of neighbors
# p: P-norm to use. 1 for sum-of-absolute, 2 for Euclidean distance

loop_this =[
	[1, 1, 0.9],
	[1, 2, 0.9],
	[2, 1, 0.9],
	[2, 2, 0.9],
]
final_scores = []
for iii in loop_this:
	print(iii)
	bpso_options = {'c1': iii[0], 'c2': iii[1], 'w':iii[2], 'k': 30, 'p':1}
	scores = evaluate_algorithm(5, dataset, bpso_options)
	final_scores.append([iii, (sum(scores) / len(scores)) * 100, scores])
	print()

for ji in final_scores:
	print(ji)