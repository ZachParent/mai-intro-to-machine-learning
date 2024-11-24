import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from tools.clustering.kmeans import KMeans
from scipy.spatial.distance import cdist
from tools.config import N_CLUSTERS

logger = logging.getLogger(__name__)

GlobalKmeansParams = {
    "n_clusters": [2, 3, 4, 5, 8, 10],
    "max_iterations": [100],
    "random_state": [1],
}

class GlobalKMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, max_iterations=300, tolerance=1e-4, random_state=None):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")

        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = {}  
        self.clusters = {}  
        self.inertia = {} 
        self.labels_ = None

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Initialize cache for distances
        distance_matrix = None

        # Initial cluster
        kmeans = KMeans(n_clusters=1, max_iterations=self.max_iterations,
                        tolerance=self.tolerance, random_state=self.random_state).fit(data)

        # Initial cluster centroids, labels & inertia
        self.centroids[1] = kmeans.centroids_
        self.clusters[1] = kmeans.labels_
        self.inertia[1] = self._compute_wcss(data, kmeans.labels_, kmeans.centroids_)

        # Initialize distance cache
        distance_matrix = cdist(data, kmeans.centroids_)

        # Expand clusters iteratively
        for k in range(2, self.n_clusters + 1):
            # Select candidate centroids (optimises runtime)
            n_candidates = int(np.sqrt(len(data)))

            # Select candidates based on minimum distance
            candidate_indices = self._select_candidates(np.min(distance_matrix, axis=1), n_candidates)

            best_centroids, best_clusters, min_inertia = None, None, float('inf')

            for idx in candidate_indices:
                # Update distance matrix for candidate centroid
                candidate_centroid = data[idx].reshape(1, -1)

                # Fit KMeans with pre-initialized centroids
                current_centroids = np.vstack((self.centroids[k-1], candidate_centroid))
                kmeans = KMeans(
                    n_clusters=k, initial_centroids=current_centroids,
                    max_iterations=self.max_iterations, tolerance=self.tolerance,
                    random_state=self.random_state
                ).fit(data)

                inertia = self._compute_wcss(data, kmeans.labels_, kmeans.centroids_)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_centroids = kmeans.centroids_
                    best_clusters = kmeans.labels_

            self.centroids[k] = best_centroids
            self.clusters[k] = best_clusters
            self.inertia[k] = min_inertia

            new_centroid = best_centroids[-1].reshape(1, -1)
            new_distances = np.linalg.norm(data - new_centroid, axis=1).reshape(-1, 1)
            distance_matrix = np.hstack((distance_matrix, new_distances))

        self.labels_ = self.clusters[self.n_clusters]
        logger.info(f"GlobalKMeans finished with {self.n_clusters} clusters")
        return self

    
    def _select_candidates(self, min_distances, n_candidates):
        """
        Select top candidates based on minimum distances using np.argpartition
        for improved efficiency.
        """
        candidate_indices = np.argpartition(min_distances, -n_candidates)[-n_candidates:]
        return candidate_indices
    

    def _compute_wcss(self, X, labels, centroids):
        """
        Vectorized WCSS computation.
        """
        return np.sum((X - centroids[labels]) ** 2)

