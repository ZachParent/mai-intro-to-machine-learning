KERNEL_PCA_PARAMS_GRID_MAP = {
    "n_components": [2, 3, 4, 10, 11, 12],
    "kernel": ["poly", "rbf"],  # RBF for non-linear data, poly for structured data
    "gamma": [
        0.1,
        1,
    ],  # For rbf and poly kernels - defines how far the influence of a single training sample reaches
}
