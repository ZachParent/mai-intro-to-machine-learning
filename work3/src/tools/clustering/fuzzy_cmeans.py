import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

FuzzyCMeansParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "fuzzyness": [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
}


class FuzzyCMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, fuzzyness: float):
        self.n_clusters = n_clusters
        self.fuzzyness = fuzzyness

    def fit(self, X):
        self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
        self.is_fitted_ = True
        return self
