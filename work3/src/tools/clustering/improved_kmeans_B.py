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
    def __init__(self, n_clusters: int, max_iterations=300, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}  # Dictionary to store centroids for each k
        self.clusters = {}  # Dictionary to store cluster labels for each k
        self.inertia = {}  # Dictionary to store WCSS for each k

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        print(f"Global K-Means: {self.n_clusters} clusters")
        
        # Initial cluster
        kmeans = KMeans(k=1, max_iterations=self.max_iterations, tolerance=self.tolerance)
        initial_centroids, initial_clusters = kmeans.fit(data)

        self.centroids[1] = initial_centroids
        self.clusters[1] = initial_clusters
        self.inertia[1] = self._compute_wcss(data, initial_clusters, initial_centroids)

        # Repeat the process until we have n_clusters centroids
        for k in range(2, self.n_clusters + 1):
            best_centroids, best_clusters = None, None
            min_inertia = float('inf')

            for i, xi in enumerate(data):
                # Add xi as a new centroid
                current_centroids = np.vstack((self.centroids[k - 1], xi))

                # Perform k-means with the new centroids
                kmeans = KMeans(k=k, centroids=current_centroids, max_iterations=self.max_iterations, tolerance=self.tolerance)
                centroids, clusters = kmeans.fit(data)

                # Compute WCSS
                inertia = self._compute_wcss(data, clusters, centroids)

                # Keep the best solution
                if inertia < min_inertia:
                    min_inertia = inertia
                    best_centroids = centroids
                    best_clusters = clusters

            # Store the best centroids and clusters for this k
            self.centroids[k] = best_centroids
            self.clusters[k] = best_clusters
            self.inertia[k] = min_inertia

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
        self.fit(data)
        final_clusters = self.clusters[self.n_clusters]
        return final_clusters
