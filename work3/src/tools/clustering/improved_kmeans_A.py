import math
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import anderson
import pandas as pd
from sklearn.decomposition import PCA

from tools.clustering.kmeans import KMeans

ImprovedKMeansAParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# G-Means

class ImprovedKMeansA(ClusterMixin, BaseEstimator):
    def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):
        self.max_depth = max_depth
        self.min_obs = min_obs
        self.random_state = random_state
        self.strictness = strictness    

        self.clusters = None
        self.centroids = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
    
        labels, centroids = self._initialize_centers(X)
        k = self.k_min



        return self.centroids, self.clusters


    def _recursiveClustering(self, data, depth, indices):
        """
        Recursively run k-means with k=2 on your data until a max_depth is reached
        or we have Gaussian clusters.
        """
        depth += 1
        if depth == self.max_depth:
            # Assign a label to all data points in this cluster
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('max_depth')
            return

        # Fit k-means with k=2
        km = KMeans(k=2)
        km.fit(data)
        centers = km.cluster_centers_

        # Project data onto the line between centers
        v = centers[1] - centers[0]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            # Can't split further; assign label
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('zero vector norm')
            return

        v_unit = v / v_norm
        x_proj = np.dot(data - centers[0], v_unit)
        x_proj = scale(x_proj)  # Standardize the projected data
        gaussian = self._gaussianCheck(x_proj)

        if gaussian:
            # Data is Gaussian; assign label
            self.cluster_labels[indices] = self.label_counter
            self.label_counter += 1
            self.stopping_criteria.append('gaussian')
            return
        else:
            # Split data and recurse
            labels = km.labels_
            for k in [0, 1]:
                current_data = data[labels == k]
                current_indices = indices[labels == k]
                if current_data.shape[0] <= self.min_obs:
                    self.cluster_labels[current_indices] = self.label_counter
                    self.label_counter += 1
                    self.stopping_criteria.append('min_obs')
                    continue
                self._recursiveClustering(current_data, depth, current_indices)

    def fit(self, data):
        """
        Fit the recursive clustering model to the data.
        """
        self.data = data
        n_samples = data.shape[0]
        indices = np.arange(n_samples)
        self.cluster_labels = {}  # Reset labels
        self.label_counter = 0    # Reset label counter
        self._recursiveClustering(data, depth=0, indices=indices)
        # Now create labels array
        labels_array = np.zeros(n_samples, dtype=int)
        for idx, label in self.cluster_labels.items():
            labels_array[idx] = label
        self.labels_ = labels_array
        

    def _gaussianCheck(self, vector):
        """
        Check whether a given input vector follows a Gaussian distribution.
        H0: vector is distributed Gaussian
        H1: vector is not distributed Gaussian
        """
        output = anderson(vector)
        if output[0] <= output[1][self.strictness]:
            return True
        else:
            return False
        

    def fit_predict(self, data):
        _, clusters = self.fit(data)
        return clusters
