from sklearn.cluster import SpectralClustering

SpectralClusteringParamsGrid = {
    "n_clusters": [2, 4, 8],
    "gamma": [0.5, 1, 2],
    "random_state": [42, 1337],
}


class SpectralClustering(SpectralClustering):
    pass
