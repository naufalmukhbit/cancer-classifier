import numpy as np
import pandas as pd

def class_to_float(data, class0, class1):
    for i in range(data.shape[0]):
        data.iloc[i, -1] = 1.0 if data.iloc[i, -1] == class1 else 0.0
    
def min_max_normalization(np_data):
    data_temp = np_data.transpose()
    minmax = []
    for i in range(data_temp.shape[0] - 1):
        minmax.append([data_temp[i].min(), data_temp[i].max()])
    for j in range(np_data.shape[0]):
        for k in range(np_data.shape[1] - 1):
            np_data[j][k] = 0 if (minmax[k][1] == minmax[k][0]) else (np_data[j][k] - minmax[k][0])/(minmax[k][1] - minmax[k][0])

def preprocess(data, class_neg, class_pos):
    class_to_float(data, class_neg, class_pos)
    data = data.to_numpy()
    min_max_normalization(data)
    
    return data