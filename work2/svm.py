from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np

class SVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel: str = 'linear', C: float = 1.0, degree: int = 3, gamma: str = 'scale'):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.svc = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma, probability=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.svc.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.svc.predict_proba(X)