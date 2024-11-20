from sklearn.cluster import SpectralClustering

SpectralClusteringParamsGrid = {
    "n_neighbors": [5, 10, 20],
    "affinity": ["nearest_neighbors", "rbf"],
    "eigen_solver": ["arpack", "lobpcg"],
    "assign_labels": ["kmeans", "cluster_qr"],
}


class SpectralClustering(SpectralClustering):
    pass
