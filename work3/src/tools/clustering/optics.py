from sklearn.cluster import OPTICS

OpticsParamsGrid = {
    "min_samples": [3, 5, 7],
}

class Optics(OPTICS):
    pass
