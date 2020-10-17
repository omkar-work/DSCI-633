import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from pprint import pprint
from copy import deepcopy

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba)!=type(None):
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions)+list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion_matrix = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        correct = self.predictions == self.actuals
        wrong = self.predictions != self.actuals
        self.acc = float(Counter(correct)[True])/len(correct)
        self.confusion_matrix = {}
        
        for label in self.classes_:
            #print(label)
            
            #for each label, start with 0. Cause of wrong vals before.
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(len(self.predictions)): 
                if self.actuals[i]==self.predictions[i]==label:     #setosa predicted as setosa
                    #print("Inside if")
                    TP += 1
                elif self.actuals[i]==label and self.actuals[i]!=self.predictions[i]: #setosa predicted as virginica
                    #print("Inside if")
                    FN += 1
                elif self.actuals[i]!= label and self.predictions[i]!=label:   #non-setosa predicted as non-setosa
                    #print("Inside if")
                    TN += 1
                elif self.actuals[i]!=label and self.predictions[i] == label:  #non-setosa predicted as setosa
                    #print("Inside if")
                    FP += 1
                
                self.confusion_matrix[label] = {"TP":TP, "TN": TN, "FP": FP, "FN": FN}
        return   
    
    def accuracy(self):
        if self.confusion_matrix==None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0

        if self.confusion_matrix==None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp+fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        label_precision = 0
                    else:
                        label_precision = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += label_precision * ratio
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        
    
        if self.confusion_matrix==None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn == 0:
                final_recall = 0
            else:
                final_recall = float(tp) / (tp + fn)
        else:
            if average == "micro":
                final_recall = self.accuracy()
            else:
                final_recall = 0
                n = len(self.classes_)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        recall_Label = 0
                    else:
                        recall_Label = float(tp) / (tp + fn)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    final_recall += recall_Label * ratio
                    
        return final_recall

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        
        
        if self.confusion_matrix==None:
            self.confusion()
        
        if target in self.classes_:
#             tp = self.confusion_matrix[target]["TP"]
#             fn = self.confusion_matrix[target]["FN"]
            prec = self.precision(target,average)
            recall = self.recall(target,average)
            if prec + recall == 0:
                f1 = 0
            else:
                f1 = 2 * ((prec * recall) / (prec + recall))
        else:
            if average == "micro":
                f1 = self.accuracy()
            else:
                f1 = 0
                n = len(self.classes_)
                for label in self.classes_:
                    prec = self.precision(label, average)
                    recall = self.recall(label,average)
                    if prec + recall == 0:
                        label_f1 = 0
                    else:
                        label_f1 = 2 * ((prec * recall) / (prec + recall))
                    if average == "macro":
                        div = 1 / len(self.classes_)
                    elif average == "weighted":
                        div = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    f1 += (label_f1 * div)
        
        return f1

    def auc(self, target):
        # compute AUC of ROC curve for each class
        # return auc = {self.classes_[i]: auc_i}, dict
        auc_target = 0
        AUC = 0
        old_FPR = 0
        if type(self.pred_proba)==type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                TPR = 0
                FPR = 0

                #print(FPR)
                #AUC = 0
                #old_FPR = FPR
                for i in order:
                    if self.actuals[i] == target:
                        tp += 1
                        fn -= 1
                        TPR = tp / (tp + fn)
                                                         
                       # TPR = self.recall(target)  #slide 8
                    else:
                        fp += 1
                        tn -= 1
                        old_FPR = FPR
                        FPR = fp / (fp + tn)
                            
                        #FPR = (fp)/(fp+tn)   #slide 8
                        
                        #AUC = 0.5 - ((FPR / 2) + (TPR / 2)) #stackoverflow formula
                        AUC += (FPR - old_FPR) * TPR
                        #AUC += AUC
                         
                    #AUC += AUC
            else:
                raise Exception("Unknown target class.")
            
            auc_target = AUC
            return auc_target