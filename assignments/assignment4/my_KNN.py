import pandas as pd
import numpy as np
from collections import Counter
from scipy import spatial

class my_KNN:

    def __init__(self, n_neighbors=5, metric="cosine", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
#         for x in X[self.X.columns]:
#             print(x)
#         #print(X[self.X.columns])

#         for x_next in self.X:
#             print(x_next)
        return



    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def minkowski_distance(self, x1, x2):
        return np.sum(((np.absolute(x1 - x2)) ** self.p)) ** 1/self.p
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.absolute(x1 - x2)) #+ np.absolute(x1[1] + x2[1]))
    
    def cosine_distance(self, x1, x2):
        #return (np.dot(x1 , x2) / (np.abs(x1) * np.abs(x2)))
#         up = np.dot(x1,x2)
#         down = np.absolute(x1) * np.absolute(x2)
        return spatial.distance.cosine(x1,x2)
#         return up/down


        
    def dist(self,x):
        # Calculate distances of training data to a single input data point (np.array)
        
        if self.metric == "minkowski":
            X_cols_vals = self.X[self.X.columns]
            distances = [self.minkowski_distance(x, x_next) for x_next in X_cols_vals.to_numpy()]
            #distances = "write your own code"


        elif self.metric == "euclidean":
            X_cols_vals = self.X[self.X.columns]
            distances = [self.euclidean_distance(x, x_next) for x_next in X_cols_vals.to_numpy()]

        elif self.metric == "manhattan":
            X_cols_vals = self.X[self.X.columns]
            distances = [self.manhattan_distance(x, x_next) for x_next in X_cols_vals.to_numpy()]
            #distances = "write your own code"


        elif self.metric == "cosine":
            X_cols_vals = self.X[self.X.columns]
            distances = [self.cosine_distance(x, x_next) for x_next in X_cols_vals.to_numpy()]
            #distances = "write your own code"
            
        else:
            raise Exception("Unknown criterion.")
            
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        
        distances = self.dist(x)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y[i] for i in k_indices]
        op = Counter(k_nearest_labels)

        return op

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")
        
        #print(X_feature)

        for x in X_feature.to_numpy():
            #print(x)
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        
        return probs


