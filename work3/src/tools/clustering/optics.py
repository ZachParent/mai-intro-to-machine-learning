import numpy as np
from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "metric": ["euclidean", "l1", "manhattan"],
    # TODO(Zach): choose just 2
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}


class Optics(OPTICS):
    pass
