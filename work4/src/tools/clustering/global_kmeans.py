import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from tools.clustering.kmeans import KMeans
from scipy.spatial.distance import cdist
from tools.config import N_CLUSTERS, RANDOM_STATE

logger = logging.getLogger(__name__)
import os
import pickle
from pathlib import Path

GlobalKmeansParams = {
    "n_clusters": N_CLUSTERS,
    "max_iterations": [100],
    "tolerance": [1e-3],
    "random_state": RANDOM_STATE,
}


class GlobalKMeans(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters: int,
        max_iterations=300,
        tolerance=1e-4,
        random_state=None,
        cache_dir=None,
    ):
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

        # Cache configuration
        self.cache_dir = cache_dir or Path(".cache/global_kmeans")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        data_hash = self._compute_data_hash(data)
        distance_matrix = np.zeros((len(data), self.n_clusters))

        # Try to load existing results from cache
        start_k = 1
        for k in range(1, self.n_clusters + 1):
            cached_results = self._load_from_cache(data_hash, k)
            if cached_results is not None:
                self.centroids[k] = cached_results["centroids"]
                self.clusters[k] = cached_results["clusters"]
                self.inertia[k] = cached_results["inertia"]
                if k > 1:
                    # Update distance matrix for cached results
                    distance_matrix = cached_results["distance_matrix"]
                start_k = k + 1
            else:
                break

        # If no cache was found, start with k=1
        if start_k == 1:
            kmeans = KMeans(
                n_clusters=1,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                random_state=self.random_state,
            ).fit(data)

            self.centroids[1] = kmeans.centroids_
            self.clusters[1] = kmeans.labels_
            self.inertia[1] = self._compute_wcss(
                data, kmeans.labels_, kmeans.centroids_
            )

            distance_matrix = cdist(data, kmeans.centroids_)

            # Cache the results
            self._save_to_cache(
                data_hash,
                1,
                {
                    "centroids": self.centroids[1],
                    "clusters": self.clusters[1],
                    "inertia": self.inertia[1],
                    "distance_matrix": distance_matrix,
                },
            )
            start_k = 2

        # Continue with remaining clusters
        for k in range(start_k, self.n_clusters + 1):
            # Adjust number of candidates based on k to optimize runtime
            base_candidates = int(np.sqrt(len(data)))

            # Reduce candidates for higher k values to balance computation
            n_candidates = max(int(base_candidates * (1 - 0.1 * (k - 1))), 10)
            candidate_indices = self._select_candidates(
                np.min(distance_matrix, axis=1), n_candidates
            )

            best_centroids, best_clusters, min_inertia = None, None, float("inf")

            for idx in candidate_indices:
                candidate_centroid = data[idx].reshape(1, -1)
                current_centroids = np.vstack(
                    (self.centroids[k - 1], candidate_centroid)
                )

                kmeans = KMeans(
                    n_clusters=k,
                    initial_centroids=current_centroids,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    random_state=self.random_state,
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

            # Cache the results
            self._save_to_cache(
                data_hash,
                k,
                {
                    "centroids": self.centroids[k],
                    "clusters": self.clusters[k],
                    "inertia": self.inertia[k],
                    "distance_matrix": distance_matrix,
                },
            )

        return self

    def _select_candidates(self, min_distances, n_candidates):
        """
        Select top candidates based on minimum distances using np.argpartition
        for improved efficiency.
        """
        candidate_indices = np.argpartition(min_distances, -n_candidates)[
            -n_candidates:
        ]
        return candidate_indices

    def _compute_wcss(self, X, labels, centroids):
        """
        Vectorized WCSS computation.
        """
        return np.sum((X - centroids[labels]) ** 2)

    def fit_predict(self, data):
        """
        Fit the model and return cluster labels.
        """
        self.fit(data)
        final_clusters = self.clusters[self.n_clusters]
        return final_clusters

    def clear_cache(self):
        """
        Clear all cached results.
        """
        try:
            for cache_file in self.cache_dir.glob("cache_*.pkl"):
                cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")

    def _get_cache_key(self, data_hash, k):
        """
        Generate a unique cache key based on data and parameters.
        """
        params_str = f"{k}_{self.max_iterations}_{self.tolerance}_{self.random_state}"
        return f"cache_{data_hash}_{params_str}.pkl"

    def _compute_data_hash(self, data):
        """
        Compute a hash of the input data for cache identification.
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        return hash(data.tobytes())

    def _load_from_cache(self, data_hash, k):
        """
        Try to load results from cache.
        """
        cache_file = self.cache_dir / self._get_cache_key(data_hash, k)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_results = pickle.load(f)
                return cached_results
            except Exception as e:
                print(f"Warning: Failed to load cache for k={k}: {e}")
        return None

    def _save_to_cache(self, data_hash, k, results):
        """
        Save results to cache.
        """
        cache_file = self.cache_dir / self._get_cache_key(data_hash, k)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Warning: Failed to save cache for k={k}: {e}")
