import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, ClusterMixin


KMeansParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 8, 10],
    "max_iterations": [100, 300, 500],
    "tolerance": [1e-5, 1e-4, 1e-3],
    "random_state": [1, 2, 3, 4, 5],
}

class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=3, max_iterations=300, tolerance=1e-4, random_state=None, initial_centroids=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = initial_centroids
        self.labels_ = None

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        n_samples, _ = data.shape

        if self.centroids is None:
            if self.random_state is not None:
                random.seed(self.random_state)
            self.centroids = data[random.sample(range(n_samples), self.n_clusters)]
        else:
            self.centroids = self.centroids = np.array(self.centroids)

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
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids
            self.labels_ = clusters

        return self


    def _assign_clusters(self, data):
        distances = np.array([[euclidean(point, centroid) for centroid in self.centroids] for point in data])
        return np.argmin(distances, axis=1)
