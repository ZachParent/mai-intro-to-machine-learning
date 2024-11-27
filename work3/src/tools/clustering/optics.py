import numpy as np
from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["auto", "ball_tree"],
    "min_samples": [5, 10, 20],
    "xi": [0.01, 0.05, 0.1],
    "min_cluster_size": [5, 10, 20]
}


class Optics(OPTICS):
    pass
