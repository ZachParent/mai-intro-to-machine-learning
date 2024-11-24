from .kmeans import KMeans, KMeansParamsGrid
from .gmeans import GMeans, GMeansParamsGrid
from .global_kmeans import GlobalKMeans, GlobalKmeansParams
from .fuzzy_cmeans import FuzzyCMeans, FuzzyCMeansParamsGrid
from .optics import Optics, OpticsParamsGrid
from .spectral_clustering import SpectralClustering, SpectralClusteringParamsGrid

# Move maps here
MODEL_MAP = {
    "kmeans": KMeans,
    "fuzzy_cmeans": FuzzyCMeans,
    "improved_kmeans_A": ImprovedKMeansA,
    "improved_kmeans_B": ImprovedKMeansB,
    "optics": Optics,
    "spectral_clustering": SpectralClustering,
}

PARAMS_GRID_MAP = {
    "kmeans": KMeansParamsGrid,
    "fuzzy_cmeans": FuzzyCMeansParamsGrid,
    "improved_kmeans_A": ImprovedKMeansAParamsGrid,
    "improved_kmeans_B": ImprovedKMeansBParamsGrid,
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
