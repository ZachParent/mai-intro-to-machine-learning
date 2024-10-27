from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from tools.voting import VotingFunc
from tools.distance import DistanceFunc

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

    def set_weights(self, weights: np.ndarray):
        """Set the weights for the features."""
        self.weights = weights

    def fit(self, X: np.ndarray[np.ndarray[np.number]], y: np.ndarray[np.integer]):
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - training data
            y: np.ndarray[np.integer] - training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray[np.ndarray[np.number]]) -> np.ndarray[np.ndarray[np.number]]:
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.ndarray[np.number]] - predicted probabilities
        """
        pass

    def predict(self, X: np.ndarray[np.ndarray[np.number]]) -> np.ndarray[np.integer]:
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.integer] - predicted classes
        """
        predictions = []
        for x in np.array(X):
            distances = [
                self.distance_func(x_train, self.weights * x)
                for x_train in self.X_train
            ]
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
