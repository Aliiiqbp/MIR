import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish

class SVM:
    def __init__(self, c, kernel, gamma):
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.svc = SVC(C=self.c, gamma=self.gamma, kernel=self.kernel)
        
    def fit(self, x, y):
        self.svc.fit(x, y)

    def predict(self, x):
        return self.svc.predict(x)
    
    def score(self, x, y):
        return self.svc.score(x, y)
    