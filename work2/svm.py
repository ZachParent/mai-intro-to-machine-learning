from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np

class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom SVM classifier that uses scikit-learn's SVC with different kernel types.
    """

    def __init__(self, kernel: str = 'linear', C: float = 1.0, degree: int = 3, gamma: str = 'scale'):
        """
        input:
            kernel: str - the kernel type to be used in the SVM ('linear', 'poly', 'rbf', 'sigmoid', etc.)
            C: float - regularization parameter (default is 1.0)
            degree: int - degree of the polynomial kernel (used if kernel='poly')
            gamma: str - kernel coefficient for 'rbf', 'poly', and 'sigmoid'. ('scale' or 'auto')
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.svc = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma, probability=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model according to the given training data.

        input:
            X: np.ndarray[np.ndarray[np.number]] - training data
            y: np.ndarray[np.integer] - training labels
        """
        self.svc.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.integer] - predicted classes
        """
        return self.svc.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for each class for the input samples.

        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.number] - predicted class probabilities
        """
        return self.svc.predict_proba(X)
