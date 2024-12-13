from sklearn.decomposition import KernelPCA as SklearnKernelPCA

KERNEL_PCA_PARAMS_GRID_MAP = {
    "n_components": [2, 3],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
}

class KernelPCA(SklearnKernelPCA):
    pass
