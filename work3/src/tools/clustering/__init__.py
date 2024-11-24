from .kmeans import KMeans, KMeansParamsGrid
from .gmeans import GMeans, GMeansParamsGrid
from .global_kmeans import GlobalKMeans, GlobalKmeansParams
from .fuzzy_cmeans import FuzzyCMeans, FuzzyCMeansParamsGrid
from .optics import Optics, OpticsParamsGrid
from .spectral_clustering import SpectralClustering, SpectralClusteringParamsGrid

MODEL_MAP = {
    "kmeans": KMeans,
    "fuzzy_cmeans": FuzzyCMeans,
    "gmeans": GMeans,
    "global_kmeans": GlobalKMeans,
    "optics": Optics,
    "spectral_clustering": SpectralClustering,
}

PARAMS_GRID_MAP = {
    "kmeans": KMeansParamsGrid,
    "fuzzy_cmeans": FuzzyCMeansParamsGrid,
    "gmeans": GMeansParamsGrid,
    "global_kmeans": GlobalKmeansParams,
    "optics": OpticsParamsGrid,
    "spectral_clustering": SpectralClusteringParamsGrid,
}

__all__ = [
    "KMeans",
    "KMeansParamsGrid",
    "GMeans",
    "GMeansParamsGrid",
    "GlobalKMeans",
    "GlobalKmeansParams",
    "FuzzyCMeans",
    "FuzzyCMeansParamsGrid",
    "Optics",
    "OpticsParamsGrid",
    "SpectralClustering",
    "SpectralClusteringParamsGrid",
    "MODEL_MAP",
    "PARAMS_GRID_MAP",
]
