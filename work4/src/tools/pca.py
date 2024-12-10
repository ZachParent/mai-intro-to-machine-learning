from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

# TODO: implement our own PCA
class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
