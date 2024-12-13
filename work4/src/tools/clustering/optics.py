import numpy as np
from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "metric": ["euclidean"],
    "algorithm": ["auto",],
    "min_samples": [5, 10, 20],
    "xi": [0.01, 0.05],
    "min_cluster_size": [20],
}


class Optics(OPTICS):
    pass
