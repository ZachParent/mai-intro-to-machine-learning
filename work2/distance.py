import typing
import pandas as pd


DistanceFunc = typing.Callable[[pd.Series, pd.Series], float]
"""A distance function takes in two rows and returns a distance."""


def manhattan_distance(row1: pd.Series, row2: pd.Series) -> float:
    """
    Calculates the Manhattan distance between two rows.
    """
    # TODO: implement
    pass


def euclidean_distance(row1: pd.Series, row2: pd.Series) -> float:
    """
    Calculates the Euclidean distance between two rows.
    """
    # TODO: implement
    pass


# TODO: implement another distance metric
