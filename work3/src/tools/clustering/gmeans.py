import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import anderson
from sklearn.preprocessing import scale
import pandas as pd
from tools.clustering.kmeans import KMeans
from sklearn.decomposition import PCA

# Parameter Grid for G-Means (if using grid search)
GMeansParamsGrid = {
    "strictness": [0, 1, 2, 3, 4],     
    "min_obs": [1, 5, 10],          
    "max_depth": [5, 10, 15],        
}

class GMeans(ClusterMixin, BaseEstimator):
=======
    def __init__(self, min_obs=10, max_depth=10, strictness=2):
        self.min_obs = min_obs
        self.max_depth = max_depth
        if strictness not in range(5):
            raise ValueError("strictness parameter must be an integer from 0 to 4")
        self.strictness = strictness

    def fit(self, data):
        """
        Fit the G-Means clustering algorithm on the input data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Initialize with one cluster
        kmeans = KMeans(n_clusters=1, max_iterations=300)
        centroids, clusters = kmeans.fit(data)

        if centroids is None or clusters is None:
            raise ValueError("KMeans fit method did not return centroids and clusters")

        self.centroids_ = centroids
        self.clusters_ = clusters

        # Start recursive splitting
        self._recursive_split(data, clusters, depth=0)

        return self

    def _recursive_split(self, data, clusters, depth):
        """
        Recursively split clusters and test for Gaussianity.
        """
        # Stop splitting if depth or cluster count limits are reached
        if depth >= self.max_depth:
            return

        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            cluster_data = data[clusters == cluster_id]

            # Skip if cluster is too small
            if len(cluster_data) < max(self.min_obs, 2):  # Require at least 2 points
                continue

            # Apply KMeans with k=2 to split the cluster
            sub_kmeans = KMeans(n_clusters=2, max_iterations=300)
            sub_centroids, sub_clusters = sub_kmeans.fit(cluster_data)

            if sub_centroids is None or sub_clusters is None:
                continue

            # Test Gaussianity
            is_gaussian = self._test_gaussianity(cluster_data)

            if is_gaussian:
                continue

            # Accept the split
            global_cluster_start = len(self.centroids_)
            self.centroids_ = np.vstack([self.centroids_, sub_centroids])
            sub_clusters += global_cluster_start
            clusters[clusters == cluster_id] = sub_clusters

            # Recursively split each sub-cluster
            self._recursive_split(data, clusters, depth + 1)

    def _test_gaussianity(self, data):
        """
        Test if the data follows a Gaussian distribution using the Anderson-Darling test.
        """
        pca = PCA(n_components=1)
        projected_data = pca.fit_transform(data).flatten()

        test_result = anderson(projected_data)

        # Compare test statistic with critical values
        return test_result.statistic < test_result.critical_values[self.strictness]

    def fit_predict(self, data):
        """
        Fit the model and return cluster labels.
        """
        self.fit(data)
        return self.clusters_