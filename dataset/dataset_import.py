import os

import numpy as np
import pandas as pd

from .preprocessing import preprocess

# def import_data():
#     train_data = 'x'
#     test_data = 'y'

def breast_cancer():
    breast_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'BreastCancer/breastCancer_train.data'), header = None)
    breast_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'BreastCancer/breastCancer_test.data'), header = None)
    
    breast_train_array = preprocess(breast_train, 'non-relapse', 'relapse')
    breast_test_array = preprocess(breast_test, 'non-relapse', 'relapse')

    return breast_train_array, breast_test_array

def colon_tumor():
    colon_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Colon Tumor/colonTumor.data'), header = None)
    
    colon_data_array = preprocess(colon_data, 'negative', 'positive')

    return colon_data_array, np.array([])

def lung_cancer():
    lung_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'LungCancer/lungCancer_train.data'), header = None)
    lung_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'LungCancer/lungCancer_test.data'), header = None)
    
    lung_train_array = preprocess(lung_train, 'Mesothelioma', 'ADCA')
    lung_test_array = preprocess(lung_test, 'Mesothelioma', 'ADCA')

    return lung_train_array, lung_test_array

def ovarian_cancer():
    ovarian_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Ovarian/ovarian_61902.data'), header = None)
    
    ovarian_data_array = preprocess(ovarian_data, 'Normal', 'Cancer')

    return ovarian_data_array, np.array([])

def prostate_cancer():
    prostate_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'prostate/prostate_TumorVSNormal_train.data'), header = None)
    prostate_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'prostate/prostate_TumorVSNormal_test.data'), header = None)
    
    prostate_train_array = preprocess(prostate_train, 'Normal', 'Tumor')
    prostate_test_array = preprocess(prostate_test, 'Normal', 'Tumor')

    return prostate_train_array, prostate_test_array