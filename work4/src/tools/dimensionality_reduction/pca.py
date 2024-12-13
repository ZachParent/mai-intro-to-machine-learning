from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA as SklearnPCA

# TODO: choose reasonable n_components and kernel
PCA_PARAMS_GRID_MAP = {
    "n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# TODO: implement our own PCA
class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = SklearnPCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
