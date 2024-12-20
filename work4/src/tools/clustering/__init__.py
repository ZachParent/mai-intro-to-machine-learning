from .global_kmeans import GlobalKMeans, GLOBAL_KMEANS_PARAMS_MAP
from .optics import Optics, OPTICS_PARAMS_MAP

CLUSTERING_MODEL_MAP = {
    "global_kmeans": GlobalKMeans,
    "optics": Optics,
}

CLUSTERING_PARAMS_MAP = {
    "global_kmeans": GLOBAL_KMEANS_PARAMS_MAP,
    "optics": OPTICS_PARAMS_MAP,
}

__all__ = [
    "CLUSTERING_MODEL_MAP",
    "CLUSTERING_PARAMS_MAP",
]
