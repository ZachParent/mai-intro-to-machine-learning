from typing import Callable, List
import pandas as pd


VotingFunc = Callable[[List[pd.Series], List[float], List[int]], int]
"""A voting function that takes in a list of rows, a list of distances, and a
list of classes and returns a class."""


def majority_class_vote(
    rows: List[pd.Series],
    distances: List[float],
    classes: List[int],
) -> int:
    """
    Returns the class that appears most frequently in the list of rows.

    input:
        rows: list[pd.Series] - list of rows
        distances: list[float] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass


def inverse_distance_weighted_vote(
    rows: List[pd.Series],
    distances: List[float],
    classes: List[int],
) -> int:
    """
    Returns the class that has the smallest sum of inverse distances to the
    rows.

    input:
        rows: list[pd.Series] - list of rows
        distances: list[float] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass


def shepards_work_vote(
    rows: List[pd.Series],
    distances: List[float],
    classes: List[int],
) -> int:
    """
    Returns the class that has the smallest sum of squared distances to the
    rows, using an exponential instead of an inverse distance scheme.

    input:
        rows: list[pd.Series] - list of rows
        distances: list[float] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass
