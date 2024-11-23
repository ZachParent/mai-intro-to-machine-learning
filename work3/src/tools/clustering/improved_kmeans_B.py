import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import euclidean
from tools.clustering.kmeans import KMeans
from scipy.spatial.distance import cdist

ImprovedKMeansBParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# Global K-Means

class ImprovedKMeansB(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, max_iterations = 300, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.centroids = {} # cluster centres
        self.clusters = {} # labels
        self.intertia = {} # WCSS

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        verbose = True

        # Initial cluster
        kmeans = KMeans(
            k=1,
            centroids=None,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        initial_centroids, initial_clusters = kmeans.fit(data)

        self.centroids[1] = initial_centroids
        self.clusters[1] = initial_clusters
        self.intertia[1] = self._compute_wcss(data,  self.clusters[1], self.centroids[1])

        # Repeat the process until we have n_clusters centroids
        for k in range(2, self.n_clusters + 1):
            if verbose:
                print(f'Solving {k}-means')

            self.intertia[k] = float('inf')
            for i, xi in enumerate(data):
                current_centroids = np.vstack((self.centroids[k-1], xi))

                kmeans = KMeans(k=k, centroids=current_centroids, tolerance=self.tolerance)
                centroids, clusters = kmeans.fit(data)
                kmeans_interia = self._compute_wcss(data, clusters, centroids)

                if(kmeans_interia < self.intertia[k]):
                    self.intertia[k] = kmeans_interia
                    self.centroids[k] = centroids
                    self.clusters[k] = clusters

        print(f"Clusters: {self.clusters}")
        print(f"Centroids: {self.centroids}")

        return self.centroids, self.clusters

    def _compute_wcss(self, X, labels, centroids):
        """Compute the Within-Cluster Sum of Squares (WCSS)."""
        wcss = 0
        for idx in range(len(centroids)):
            cluster_points = X[labels == idx]
            if len(cluster_points) > 0:
                centroid = centroids[idx]
                wcss += np.sum((cluster_points - centroid) ** 2)
        return wcss

    def fit_predict(self, data):
        _, clusters = self.fit(data)
        return clusters
