import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from tools.clustering.kmeans import KMeans
from scipy.spatial.distance import cdist

GlobalKmeansParams = {
    "n_clusters": [2, 3, 4, 5, 8, 10],
    "max_iterations": [100],
    "random_state": [1],
}

class GlobalKMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, max_iterations=300, tolerance=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = {}  
        self.clusters = {}  
        self.inertia = {} 

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        
        # Initial cluster
        kmeans = KMeans(k=1, max_iterations=self.max_iterations, tolerance=self.tolerance)
        initial_centroids, initial_clusters = kmeans.fit(data)

        self.centroids[1] = initial_centroids
        self.clusters[1] = initial_clusters
        self.inertia[1] = self._compute_wcss(data, initial_clusters, initial_centroids)

        # Repeat the process until we have n_clusters centroids
        for k in range(2, self.n_clusters + 1):
            # Fast Global k-means optimization: Calculate distances to existing centroids
            distances = cdist(data, self.centroids[k-1])
            min_distances = np.min(distances, axis=1)
            
            # Select top N candidates (e.g., sqrt(n) points with highest min_distances)
            n_candidates = int(np.sqrt(len(data)))
            candidate_indices = np.argsort(min_distances)[-n_candidates:]
            
            best_centroids, best_clusters = None, None
            min_inertia = float('inf')

            # Only try the most promising candidates
            for idx in candidate_indices:
                current_centroids = np.vstack((self.centroids[k - 1], data[idx]))
                
                # Perform k-means with the new centroids
                kmeans = KMeans(k=k, centroids=current_centroids, 
                              max_iterations=self.max_iterations, tolerance=self.tolerance, random_state=self.random_state)
                centroids, clusters = kmeans.fit(data)

                inertia = self._compute_wcss(data, clusters, centroids)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_centroids = centroids
                    best_clusters = clusters

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
