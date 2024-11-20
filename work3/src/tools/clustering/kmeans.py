import random
import numpy as np
import pandas as pd
from tools.clustering.distance import ManhattanDistance, EuclideanDistance, ChebyshevDistance

KMeansParamsGrid = {
    "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "distance-metrics": [ManhattanDistance(), EuclideanDistance(), ChebyshevDistance()]
}


class KMeans:
    def __init__(self, **kwargs):
        self.k = kwargs.get("k", 3)
        self.distance_metric = kwargs.get("distance_metric", EuclideanDistance())
        self.max_iterations = kwargs.get("max_iterations", 300)
        self.tolerance = kwargs.get("tolerance", 1e-4)
        self.centroids = None

    def fit(self, data):
        data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        n_samples, n_features = data.shape
        self.centroids = data[random.sample(range(n_samples), self.k)]

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

        return self.centroids, clusters

    def fit_predict(self, data):
        _, clusters = self.fit(data)
        return clusters

    def _assign_clusters(self, data):
        distances = np.array(
            [[self.distance_metric(point, centroid) for centroid in self.centroids] for point in data]
        )
        return np.argmin(distances, axis=1)
