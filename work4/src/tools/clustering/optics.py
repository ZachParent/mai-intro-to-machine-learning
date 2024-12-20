from sklearn.cluster import OPTICS

OPTICS_PARAMS_MAP = {
    "mushroom": {
        "metric": "euclidean",
        "algorithm": "ball_tree",
        "min_samples": 10,
        "xi": 0.1,
        "min_cluster_size": 5,
    },
    "vowel": {
        "metric": "manhattan",
        "algorithm": "ball_tree",
        "min_samples": 20,
        "xi": 0.1,
        "min_cluster_size": 10,
    },
}


class Optics(OPTICS):
    pass
