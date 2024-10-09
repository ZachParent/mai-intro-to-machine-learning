from typing import Callable
import numpy as np
from voting import VotingFunc
from distance import DistanceFunc

KNNClassifier = Callable[
    [np.ndarray[np.ndarray[np.number], np.ndarray[np.integer]], int]
]


def train_knn(
    X: np.ndarray[np.ndarray[np.number]],
    y: np.ndarray[np.integer],
    k: int,
    distance_func: DistanceFunc,
    voting_func: VotingFunc,
    weights: np.ndarray[np.number],
) -> KNNClassifier:
    """
    Trains a KNN classifier.

    input:
        X: np.ndarray[np.ndarray[np.number]] - training data
        y: np.ndarray[np.integer] - training labels
        k: int - number of neighbors
        distance_func: DistanceFunc - distance function
        voting_func: VotingFunc - voting function
        weights: np.ndarray[np.number] - weights for each feature

    output:
        callable[[pd.Series], int] - classifier
            Takes in a row to classify and returns a class
    """

    def classifier(row: np.ndarray[np.number]) -> int:
        # TODO: implement
        pass

    return classifier
