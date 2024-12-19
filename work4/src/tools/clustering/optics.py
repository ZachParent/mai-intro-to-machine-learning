from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["ball_tree"],
    "min_samples": [10, 20],
    "xi": [0.1],
    "min_cluster_size": [5, 10],
}


class Optics(OPTICS):
    pass
