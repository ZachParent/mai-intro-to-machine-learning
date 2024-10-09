import pandas as pd


def train_knn(
    data: pd.DataFrame,
    k: int,
    distance_func: callable[[pd.Series, pd.Series], float],
    voting_func: callable[[list[pd.Series], list[int]], int],
    weights: list[float],
) -> callable[[pd.Series], int]:
    """
    Trains a KNN classifier.

    input:
        data: pd.DataFrame - training data
        k: int - number of neighbors
        distance_func: callable[[pd.Series, pd.Series], float] - distance function
            Takes in a 2 rows to compare and returns a distance
        voting_func: callable[[list[pd.Series], list[int]], int] - voting function
            Takes in a list of rows and a list of the corresponding classes and
            returns a class from one of the rows
        weights: list[float] - list of weights for each feature

    output:
        callable[[pd.Series], int] - classifier
            Takes in a row to classify and returns a class
    """
    # TODO: implement
    pass
