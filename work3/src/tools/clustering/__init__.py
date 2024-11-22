from .kmeans import KMeans, KMeansParamsGrid
from .improved_kmeans_A import ImprovedKMeansA, ImprovedKMeansAParamsGrid
from .improved_kmeans_B import ImprovedKMeansB, ImprovedKMeansBParamsGrid
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
    "ImprovedKMeansA",
    "ImprovedKMeansAParamsGrid",
    "ImprovedKMeansB",
    "ImprovedKMeansBParamsGrid",
    "FuzzyCMeans",
    "FuzzyCMeansParamsGrid",
    "Optics",
    "OpticsParamsGrid",
    "SpectralClustering",
    "SpectralClusteringParamsGrid",
    "MODEL_MAP",
    "PARAMS_GRID_MAP",
]
