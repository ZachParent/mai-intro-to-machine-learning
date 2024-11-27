from sklearn.cluster import SpectralClustering
from tools.config import RANDOM_STATE

SpectralClusteringParamsGrid = {
    "n_neighbors": [5, 10, 20],
    "affinity": ["nearest_neighbors", "rbf"],
    "eigen_solver": ["arpack", "lobpcg"],
    "assign_labels": ["kmeans", "cluster_qr"],
    "random_state": RANDOM_STATE,
}


class SpectralClustering(SpectralClustering):
    pass
