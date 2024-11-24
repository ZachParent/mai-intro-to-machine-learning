import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, ClusterMixin
from tools.config import N_CLUSTERS, RANDOM_STATE

KMeansParamsGrid = {
    "n_clusters": N_CLUSTERS,
    "max_iterations": [100, 300, 500],
    "tolerance": [1e-5, 1e-4, 1e-3],
    "random_state": RANDOM_STATE,
}

class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=2, max_iterations=300, tolerance=1e-4, random_state=None, initial_centroids=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids_ = initial_centroids
        self.labels_ = None

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        n_samples, _ = data.shape

        if self.centroids_ is None:
            if self.random_state is not None:
                random.seed(self.random_state)
            self.centroids_ = data[random.sample(range(n_samples), self.n_clusters)]
        else:
            self.centroids_ = np.array(self.centroids_)

        for _ in range(self.max_iterations):
            # Assign clusters
            clusters = self._assign_clusters(data)

            # Update centroids
            new_centroids = np.array([
                data[clusters == i].mean(axis=0) if len(data[clusters == i]) > 0 else
                data[random.sample(range(n_samples), 1)][0]
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids_) < self.tolerance):
                break

            self.centroids_ = new_centroids
            self.labels_ = clusters

        return self


    def _assign_clusters(self, data):
        distances = np.array([[euclidean(point, centroid) for centroid in self.centroids_] for point in data])
        return np.argmin(distances, axis=1)
