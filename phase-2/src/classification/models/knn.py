import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin


class KNN(BaseEstimator, ClassifierMixin):
    
    def __init__(self, k: int):
        self.k = k
        pass

    def fit(self, x, y, **fit_params):
        self.x_train = x
        self.y_train = y
        return self

    def predict(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        
        threeSums = np.sum(np.square(x)[:,np.newaxis,:], axis=2) - 2 * x.dot(self.x_train.T) + np.sum(np.square(self.x_train), axis=1)
        dists = np.sqrt(threeSums)
        
        
        
        pass

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            closest_y = [self.y_train[j] for j in np.argsort(dists[i])[0:k]]
            y_pred[i] = Counter(closest_y).most_common(1)[0][0]
        
        return y_pred
