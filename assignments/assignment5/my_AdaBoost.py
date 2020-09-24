import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
from sklearn.tree import DecisionTreeClassifier

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # Compute alpha for estimator i
            non_zero_denom = 1e-10
            log_k = np.log(1 - 1.0 / k)
            log_error = np.log((1 - error) / (error + non_zero_denom))
            self.alpha.append(log_k + log_error)

            # Update wi
            for i in range(len(diffs)):
                if(diffs[i]):
                    w[i] *= np.exp(self.alpha[-1])
                    #w[i] = w[i]
                else:
                    #w[i] *= np.exp(self.alpha[-1] * diffs[i])
                    w[i] = w[i]
            
            
            #normalisation of w
            w = w / np.sum(w)    
            #print(np.sum(w))
            #w = "write your own code"

        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        

        for label in self.classes_:
            running_total = 0
            for i in range(self.n_estimators):
                running_total += ((self.alpha[i]) * (self.estimators[i].predict(X) == label))
            probs[label] = (running_total)
            
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs