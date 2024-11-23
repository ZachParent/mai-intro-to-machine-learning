import math
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import anderson
import pandas as pd
from sklearn.preprocessing import scale
from tools.clustering.kmeans import KMeans

# TODO: Update accordingly
GMeansParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class GMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, min_obs=1, max_depth=10, random_state=None, strictness=4):
        self.n_clusters = n_clusters
        self.min_obs = min_obs
        self.max_depth = max_depth
        self.random_state = random_state
        self.strictness = strictness

    def _recursiveClustering(self, data, depth, indices):
        depth += 1
        print(f"Recursion Depth: {depth}, Data Shape: {data.shape}, Indices: {len(indices)}")
        
        if depth == self.max_depth or len(data) <= self.min_obs:
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('max_depth' if depth == self.max_depth else 'min_obs')
            return

        # Fit KMeans with k=2
        km = KMeans(k=2, random_state=self.random_state)
        km.fit(data)
        centers = km.centroids
        labels = km.clusters

        # Calculate projection vector
        v = centers[1] - centers[0]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('zero vector norm')
            return

        v_unit = v / v_norm
        x_proj = np.dot(data - centers[0], v_unit)
        x_proj = scale(x_proj)  # Standardize the projection

        # Check for Gaussianity
        if self._gaussianCheck(x_proj):
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('gaussian')
            return

        # If not Gaussian, recursively cluster
        for k in [0, 1]:
            mask = (labels == k)
            current_data = data[mask]
            current_indices = indices[mask]
            if len(current_data) <= self.min_obs:
                self.cluster_labels[current_indices] = self.label_counter
                self.label_counter += 1
                self.stopping_criteria.append('min_obs')
                continue
            self._recursiveClustering(current_data, depth, current_indices)

    def _gaussianCheck(self, vector):
        if len(vector) < 10:  # Minimum size for Anderson-Darling test
            return False
        output = anderson(vector)
        return output[0] <= output[1][self.strictness]

    def fit(self, data):
        data = np.array(data)
        n_samples = data.shape[0]
        self.cluster_labels = np.full(n_samples, -1, dtype=int)  # Initialize with -1
        self.label_counter = 0
        self.stopping_criteria = []
        self._recursiveClustering(data, depth=0, indices=np.arange(n_samples))
        self.labels_ = self.cluster_labels
        return self

    def fit_predict(self, data):
        self.fit(data)
        return self.labels_