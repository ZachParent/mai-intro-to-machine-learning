import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import euclidean
from tools.clustering.kmeans import KMeans

# Global K-Means

ImprovedKMeansBParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class ImprovedKMeansB(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, n_init: int = 2):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.centroids = None
        self.clusters = None
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        best_overall_score = float('inf')
        best_overall_centroids = None
        best_overall_clusters = None

        for k in range(1, self.n_clusters + 1):
            best_score = float('inf')
            best_centroids = None
            best_clusters = None

            # Run KMeans n_init times for each k
            for _ in range(self.n_init):
                kmeans = KMeans(k=k)
                centroids, clusters = kmeans.fit(data)

                # Compute WCSS (Within-Cluster Sum of Squares)
                score = sum(
                    np.sum((data[clusters == j] - centroids[j]) ** 2)
                    for j in range(k)
                )

                # If the score is better (lower cost), update the best clustering
                if score < best_score:
                    best_score = score
                    best_centroids = centroids
                    best_clusters = clusters

            # Update overall best if this clustering is better
            if best_score < best_overall_score:
                best_overall_score = best_score
                best_overall_centroids = best_centroids
                best_overall_clusters = best_clusters

        self.centroids = best_overall_centroids
        self.clusters = best_overall_clusters

        return self.centroids, self.clusters
    
    def fit_predict(self, data):
        _, clusters = self.fit(data)
        return clusters