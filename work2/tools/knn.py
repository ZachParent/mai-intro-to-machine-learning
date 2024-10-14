from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from typing import Callable
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


def run_knn(
    knn: KNNClassifier, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run KNN algorithm on the given train and test datasets.

    Args:
        knn (KNNClassifier): The KNN model to use.
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.
        target_col (str): The name of the target column.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing predictions and actual values.
    """
    knn.fit(train_df.drop(target_col, axis=1), train_df[target_col])
    preds = knn.predict(test_df.drop(target_col, axis=1))
    actuals = test_df[target_col]
    return (preds, actuals)


def cross_validate_knn(
    knn: KNNClassifier,
    train_dfs: list[pd.DataFrame],
    test_dfs: list[pd.DataFrame],
    target_col: str,
    score_func: Callable[[np.ndarray, np.ndarray], any],
) -> np.ndarray:
    """
    Perform cross-validation for the KNN model.

    Args:
        knn (KNNClassifier): The KNN model to use.
        train_dfs (list[pd.DataFrame]): List of training datasets.
        test_dfs (list[pd.DataFrame]): List of testing datasets.
        target_col (str): The name of the target column.
        score_func (Callable[[np.ndarray, np.ndarray], float]): The scoring function to use.

    Returns:
        np.ndarray: An array of scores from cross-validation.
    """
    scores = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        preds, actuals = run_knn(knn, train_df, test_df, target_col)
        scores.append(score_func(actuals, preds))
    return np.array(scores)
