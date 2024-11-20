import numpy as np
from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "min_samples": [3, 5, 10],
    "max_eps": [np.inf, 2.0],
    "cluster_method": ["dbscan", "xi"],
}


class Optics(OPTICS):
    pass
