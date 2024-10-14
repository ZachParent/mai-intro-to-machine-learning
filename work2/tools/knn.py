from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
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
        self.X_train = np.array(X)
        self.y_train = np.array(y)
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
        for x in np.array(X):
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

def run_knn(knn, train_df, test_df, target_col: str):
    knn.fit(train_df.drop(target_col, axis=1), train_df[target_col])
    preds = knn.predict(test_df.drop(target_col, axis=1))
    actuals = test_df[target_col]
    return (preds, actuals)

def cross_validate_knn(knn, train_dfs, test_dfs, target_col: str):
    f1_scores = []
    for train_df, test_df in zip(train_dfs, test_dfs):  
        preds, actuals = run_knn(knn, train_df, test_df, target_col)
        f1_scores.append(f1_score(actuals, preds, average="weighted"))
    return np.array(f1_scores)
