from typing import Callable, List
import pandas as pd
from voting import VotingFunc
from distance import DistanceFunc

KNNClassifier = Callable[[pd.Series], int]


def train_knn(
    data: pd.DataFrame,
    k: int,
    distance_func: DistanceFunc,
    voting_func: VotingFunc,
    weights: List[float],
) -> KNNClassifier:
    """
    Trains a KNN classifier.

    input:
        data: pd.DataFrame - training data
        k: int - number of neighbors
        distance_func: DistanceFunc - distance function
        voting_func: VotingFunc - voting function
        weights: list[float] - list of weights for each feature

    output:
        callable[[pd.Series], int] - classifier
            Takes in a row to classify and returns a class
    """
    # TODO: implement
