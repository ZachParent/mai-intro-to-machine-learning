from .pca import PCA, OurPCA, PCA_PARAMS_GRID_MAP
from .kernel_pca import KernelPCA, KERNEL_PCA_PARAMS_GRID_MAP

REDUCTION_METHOD_MAP = {
    "pca": OurPCA,
    "kernel_pca": KernelPCA,
}

REDUCTION_PARAMS_GRID_MAP = {
    "pca": PCA_PARAMS_GRID_MAP,
    "kernel_pca": KERNEL_PCA_PARAMS_GRID_MAP,
}

__all__ = ["PCA", "KernelPCA", "PCA_PARAMS_GRID_MAP", "KERNEL_PCA_PARAMS_GRID_MAP"]
