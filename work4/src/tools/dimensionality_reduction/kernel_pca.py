from sklearn.decomposition import KernelPCA as SklearnKernelPCA

KERNEL_PCA_PARAMS_GRID_MAP = {
    "n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
}

class KernelPCA(SklearnKernelPCA):
    pass
