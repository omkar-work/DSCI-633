import numpy as np
import pandas as pd
from scipy.linalg import svd
from copy import deepcopy
from pdb import set_trace
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

def pca(X, n_components = 5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)
    #  's' is eigen value and 'Vh' is eigen_vector

    U, s, Vh = svd(X)
    Vh_t = Vh.T
    
    principal_components = Vh_t[:, :n_components]
    return principal_components




def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector
    # Input x: 1-d np.array
    if norm == "Min-Max":
        x_norm = (x - np.min(x))/(np.max(x) - np.min(x))
    elif norm == "L1":
        x_norm = (np.abs(x) / np.sum(np.abs(x)))
    elif norm == "L2":
        x_norm = (np.abs(x) / np.sqrt(np.sum(np.square(x))))
    elif norm == "Standard_Score":
        x_norm = (x - np.mean(x) / np.std(x))
    else:
        raise Exception("Unknown normlization.")
    return x_norm



def normalize(X, norm="Min-Max", axis = 1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:,col] = vector_norm(X_norm[:,col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm





def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    
    # we need unique list of classes in y
    y_classes_ = list(set(y))
    
    #store the occurances of these classes somewhere (indices)
    y_indices = {}
    
    for y_class in y_classes_:
        for classes in y:
            indices = np.where(y == y_class)
            y_indices[y_class] = indices
    
    # y_indices
    sample = []
    
    for classes in y_classes_:
        n = np.ceil(len(y_indices[classes][0] * ratio))
        sample.append(np.random.choice(y_indices[classes][0], int(n), replace = replace))
    
    sample = np.concatenate(sample)
    


    return sample.astype(int)
