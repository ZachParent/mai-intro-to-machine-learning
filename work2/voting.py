from typing import Callable, List
import numpy as np


VotingFunc = Callable[[List[np.number], List[int]], int]
"""A voting function that takes in a list of distances, and a list of classes and returns a class."""


def majority_class_vote(
    distances: List[np.number],
    classes: List[int],
) -> int:
    """
    Returns the class that appears most frequently in the list of rows.

    input:
        distances: list[np.number] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass


def inverse_distance_weighted_vote(
    distances: List[np.number],
    classes: List[int],
) -> int:
    """
    Returns the class that has the smallest sum of inverse distances to the
    rows.

    input:
        distances: list[np.number] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass


def shepards_work_vote(
    distances: List[np.number],
    classes: List[int],
) -> int:
    """
    Returns the class that has the smallest sum of squared distances to the
    rows, using an exponential instead of an inverse distance scheme.

    input:
        distances: list[np.number] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """
    # TODO: implement
    pass
