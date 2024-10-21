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

    def predict(self, X: np.ndarray[np.ndarray[np.number]]) -> np.ndarray[np.integer]:
        """
        input:
            X: np.ndarray[np.ndarray[np.number]] - data to classify

        output:
            np.ndarray[np.integer] - predicted classes
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Computing the distance
        distances = []

        for x_train in self.X_train:
            if self.weights is not None:
                x, x_train = self.apply_weights(x, x_train, self.weights)

            distances.append(self.distance_func(x, x_train))

        # Get the closest k neighbours
        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # Pass both distances and labels
        return self.voting_func(k_nearest_distances, k_nearest_labels)
    
    def predict_proba(self, X: np.ndarray[np.ndarray[np.number]]) -> np.ndarray:
        """
        Estimate probability for each class by counting the votes from neighbors.
        Returns an array of shape (n_samples, n_classes) with probability estimates.
        """
        probas = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                if self.weights is not None:
                    x, x_train = self.apply_weights(x, x_train, self.weights)
                distances.append(self.distance_func(x, x_train))
            
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Count the occurrences of each class in the k neighbors
            class_votes = np.zeros(len(self.classes_))
            for label in k_nearest_labels:
                class_votes[label] += 1
            
            # Convert vote counts into probabilities by normalizing
            class_probabilities = class_votes / np.sum(class_votes)
            probas.append(class_probabilities)
        
        return np.array(probas)
    
    def apply_weights(self, row1: np.ndarray, row2: np.ndarray, weights: np.ndarray):
        """
        Multiplies each feature in row1 and row2 by the corresponding feature weight.
        
        input:
            row1: np.ndarray - the first feature vector
            row2: np.ndarray - the second feature vector
            weights: np.ndarray - weights for each feature
            
        output:
            Tuple of row1 and row2 with applied weights.
        """
        return row1 * weights, row2 * weights

