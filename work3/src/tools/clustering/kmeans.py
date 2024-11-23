import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

KMeansParamsGrid = {
    "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class KMeans:
    def __init__(self, **kwargs):
        self.k = kwargs.get("k", 3)
        self.max_iterations = kwargs.get("max_iterations", 300)
        self.tolerance = kwargs.get("tolerance", 1e-4)
        self.centroids = kwargs.get("initial_centroids", None)
        self.clusters = None

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        n_samples, n_features = data.shape

        if self.centroids is None:
            self.centroids = data[random.sample(range(n_samples), self.k)]
        else:
            self.centroids = self.centroids = np.array(self.centroids)

        for _ in range(self.max_iterations):
            # Assign clusters
            clusters = self._assign_clusters(data)

            # Update centroids
            new_centroids = np.array([
                data[clusters == i].mean(axis=0) if len(data[clusters == i]) > 0 else
                data[random.sample(range(n_samples), 1)][0]
                for i in range(self.k)
            ])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids
            self.clusters = clusters

        return self.centroids, self.clusters

    def fit_predict(self, data):
        _, clusters = self.fit(data)
        return clusters

    def _assign_clusters(self, data):
        distances = np.array([[euclidean(point, centroid) for centroid in self.centroids] for point in data])
        return np.argmin(distances, axis=1)
