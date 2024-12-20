from .pca import PCA, PCA_PARAMS_GRID_MAP
from .kernel_pca import KERNEL_PCA_PARAMS_GRID_MAP
from sklearn.decomposition import PCA as SklearnPCA, KernelPCA as SklearnKernelPCA

REDUCTION_METHOD_MAP = {
    "pca": PCA,
    "sklearn_pca": SklearnPCA,
    "kernel_pca": SklearnKernelPCA,
}

REDUCTION_PARAMS_GRID_MAP = {
    "pca": PCA_PARAMS_GRID_MAP,
    "sklearn_pca": PCA_PARAMS_GRID_MAP,
    "kernel_pca": KERNEL_PCA_PARAMS_GRID_MAP,
}

__all__ = [
    "PCA",
    "SklearnPCA",
    "SklearnKernelPCA",
    "PCA_PARAMS_GRID_MAP",
    "KERNEL_PCA_PARAMS_GRID_MAP",
    "REDUCTION_METHOD_MAP",
    "REDUCTION_PARAMS_GRID_MAP",
]
