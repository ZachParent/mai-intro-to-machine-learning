from .pca import PCA, PCA_PARAMS_GRID_MAP
from .kernel_pca import KERNEL_PCA_PARAMS_GRID_MAP
from .incremental_pca import INCREMENTAL_PCA_PARAMS_GRID_MAP
from sklearn.decomposition import PCA as SklearnPCA, KernelPCA as SklearnKernelPCA, IncrementalPCA as SklearnIncrementalPCA

REDUCTION_METHOD_MAP = {
    "pca": PCA,
    "sklearn_pca": SklearnPCA,
    "kernel_pca": SklearnKernelPCA,
    "incremental_pca": SklearnIncrementalPCA,
}

REDUCTION_PARAMS_GRID_MAP = {
    "pca": PCA_PARAMS_GRID_MAP,
    "sklearn_pca": PCA_PARAMS_GRID_MAP,
    "kernel_pca": KERNEL_PCA_PARAMS_GRID_MAP,
    "incremental_pca": INCREMENTAL_PCA_PARAMS_GRID_MAP,
}

__all__ = [
    "PCA",
    "SklearnPCA",
    "SklearnIncrementalPCA",
    "SklearnKernelPCA",
    "PCA_PARAMS_GRID_MAP",
    "KERNEL_PCA_PARAMS_GRID_MAP",
    "REDUCTION_METHOD_MAP",
    "REDUCTION_PARAMS_GRID_MAP",
]
