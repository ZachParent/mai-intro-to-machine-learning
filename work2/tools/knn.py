from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from .voting import VotingFunc
from .distance import DistanceFunc


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A KNN classifier.
    """

    X_train: np.ndarray[np.ndarray[np.number]]
    y_train: np.ndarray[np.integer]

    def __init__(
        self,
        k: int,
        distance_func: DistanceFunc,
        voting_func: VotingFunc,
        weights: np.ndarray[np.number],
    ):
        """
        input:
            k: int - number of neighbors
            distance_func: DistanceFunc - distance function
            voting_func: VotingFunc - voting function
            weights: np.ndarray[np.number] - weights for each feature
        """
        self.k = k
        self.distance_func = distance_func
        self.voting_func = voting_func
        self.weights = weights

    def fit(self, X: np.ndarray[np.ndarray[np.number]], y: np.ndarray[np.integer]):
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - training data
            y: np.ndarray[np.integer] - training labels
        """
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray[np.ndarray[np.number]]) -> np.ndarray[np.integer]:
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.integer] - predicted classes
        """
        predictions = []
        for x in X:
            distances = [self.distance_func(x_train, self.weights * x) for x_train in self.X_train]
            distances_and_classes = list(zip(distances, self.y_train))
            sorted_distances_and_classes = sorted(
                distances_and_classes, key=lambda dis_and_cls: dis_and_cls[0]
            )
            k_nearest_distances_and_classes = sorted_distances_and_classes[: self.k]
            k_nearest_distances = [dis for dis, _ in k_nearest_distances_and_classes]
            k_nearest_classes = [cls for _, cls in k_nearest_distances_and_classes]
            pred = self.voting_func(k_nearest_distances, k_nearest_classes)
            predictions.append(pred)
        return np.array(predictions)
