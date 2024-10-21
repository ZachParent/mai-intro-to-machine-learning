from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
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

    def condensed_nearest_neighbor(self, X: pd.DataFrame, y: pd.Series):
        self.X_reduced = pd.DataFrame(columns=X.columns)  # Initialize an empty DataFrame with the same columns as X
        self.y_reduced = pd.Series(dtype=y.dtype)  # Initialize an empty Series for y_reduced

        # Start with the first instance
        self.X_reduced = pd.concat([self.X_reduced, X.iloc[[0]]], ignore_index=True)
        self.y_reduced = pd.concat([self.y_reduced, pd.Series([y.iloc[0]])], ignore_index=True)

        for i in range(1, len(X)):
            distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), self.X_reduced.values)
            nearest_labels = self.y_reduced.iloc[np.argsort(distances[0])[:self.k]]

            # If the instance is misclassified, add it to the reduced set
            if nearest_labels.value_counts().get(y.iloc[i], 0) <= self.k / 2:
                self.X_reduced = pd.concat([self.X_reduced, X.iloc[[i]]], ignore_index=True)
                self.y_reduced = pd.concat([self.y_reduced, pd.Series([y.iloc[i]])], ignore_index=True)

        return self.X_reduced.reset_index(drop=True), self.y_reduced.reset_index(drop=True)

    def edited_nearest_neighbor(self, X: pd.DataFrame, y: pd.Series):
        # Edited Nearest Neighbor reduction
        X_reduced_list = []
        y_reduced_list = []

        for i in range(len(X)):
            distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), X.values)
            k_indices = np.argsort(distances[0])[:self.k]
            k_nearest_labels = y.iloc[k_indices]

            if k_nearest_labels.value_counts().get(y.iloc[i], 0) > self.k / 2:
                X_reduced_list.append(X.iloc[i])
                y_reduced_list.append(y.iloc[i])

        # Concatenate at once at the end
        self.X_reduced = pd.DataFrame(X_reduced_list).reset_index(drop=True)
        self.y_reduced = pd.Series(y_reduced_list).reset_index(drop=True)

        return self.X_reduced, self.y_reduced

    def drop2(self, X: pd.DataFrame, y: pd.Series):
        # Drop2 reduction
        X_reduced_list = []
        y_reduced_list = []

        for i in range(len(X)):
            distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), X.values)
            k_indices = np.argsort(distances[0])[:self.k]
            k_nearest_labels = y.iloc[k_indices]

            # Count of the nearest neighbors with the same label
            count_same_label = (k_nearest_labels == y.iloc[i]).sum()
            count_diff_label = self.k - count_same_label

            # Condition for removal: if more than half of the neighbors are different
            if count_diff_label <= count_same_label:
                X_reduced_list.append(X.iloc[i])
                y_reduced_list.append(y.iloc[i])

        # Concatenate at once at the end
        self.X_reduced = pd.DataFrame(X_reduced_list).reset_index(drop=True)
        self.y_reduced = pd.Series(y_reduced_list).reset_index(drop=True)

        return self.X_reduced, self.y_reduced