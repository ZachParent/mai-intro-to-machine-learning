import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

ImprovedKMeansAParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}


class ImprovedKMeansA(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def fit(self, X):
        self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
        self.is_fitted_ = True
        return self

