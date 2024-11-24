import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from tools.clustering.kmeans import KMeans
from scipy.spatial.distance import cdist
from tools.config import N_CLUSTERS

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
        kmeans = KMeans(n_clusters=1, max_iterations=self.max_iterations,
                        tolerance=self.tolerance, random_state=self.random_state).fit(data)

        self.centroids[1] = kmeans.centroids
        self.clusters[1] = kmeans.labels_
        min_distances = np.min(cdist(data, kmeans.centroids), axis=1)
        self.inertia[1] = self._compute_wcss(data, kmeans.labels_, kmeans.centroids)

        # Expand clusters iteratively
        for k in range(2, self.n_clusters + 1):
            n_candidates = int(np.sqrt(len(data)))
            candidate_indices = self._select_candidates(data, min_distances, n_candidates)

            best_centroids, best_clusters, min_inertia = None, None, float('inf')

            for idx in candidate_indices:
                current_centroids = np.vstack((self.centroids[k-1], data[idx]))
                kmeans = KMeans(
                    n_clusters=k, initial_centroids=current_centroids,
                    max_iterations=self.max_iterations, tolerance=self.tolerance,
                    random_state=self.random_state
                ).fit(data)

                inertia = self._compute_wcss(data, kmeans.labels_, kmeans.centroids)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_centroids = kmeans.centroids
                    best_clusters = kmeans.labels_

            self.centroids[k] = best_centroids
            self.clusters[k] = best_clusters
            self.inertia[k] = min_inertia

            # Efficiently update min_distances
            new_centroid = best_centroids[-1].reshape(1, -1)
            distances_to_new_centroid = np.linalg.norm(data - new_centroid, axis=1)
            min_distances = np.minimum(min_distances, distances_to_new_centroid)

        return self
    
    def _select_candidates(self, min_distances, n_candidates):
        """
        Select top candidates based on minimum distances using np.argpartition
        for improved efficiency.
        """
        candidate_indices = np.argpartition(min_distances, -n_candidates)[-n_candidates:]
        return candidate_indices
    

    def _compute_wcss(self, X, labels, centroids):
        """Vectorized WCSS computation."""
        return np.sum((X - centroids[labels]) ** 2)


    def fit_predict(self, data):
        self.fit(data)
        final_clusters = self.clusters[self.n_clusters]
        return final_clusters
