from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A KNN classifier.
    """

    def __init__(self, k: int, distance_func, voting_func, weights: np.ndarray = None):
        """
        input:
            k: int - number of neighbors
            distance_func: callable - function to compute distance between points
            voting_func: callable - function to vote for predicted class
            weights: np.ndarray[np.number] - weights for each feature (optional)
        """
        self.k = k
        self.distance_func = distance_func
        self.voting_func = voting_func
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - training data
            y: np.ndarray[np.integer] - training labels
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.integer] - predicted classes
        """
        predictions = []
        for x_test in X:
            # Compute distances between x_test and all training data
            distances = np.array([self.distance_func(x_test, x_train, self.weights) for x_train in self.X_train])

            # Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]

            # Get the corresponding labels for the k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Predict the class based on voting function
            predicted_class = self.voting_func(k_nearest_labels)

            predictions.append(predicted_class)

        return np.array(predictions)
