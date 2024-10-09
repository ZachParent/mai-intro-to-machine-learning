import typing
import numpy as np


DistanceFunc = typing.Callable[
    [np.ndarray[np.number], np.ndarray[np.number]], np.number
]
"""A distance function takes in two rows and returns a distance."""


def manhattan_distance(
    row1: np.ndarray[np.number], row2: np.ndarray[np.number]
) -> np.number:
    """
    Calculates the Manhattan distance between two rows.
    """
    # TODO: implement
    pass


def euclidean_distance(
    row1: np.ndarray[np.number], row2: np.ndarray[np.number]
) -> np.number:
    """
    Calculates the Euclidean distance between two rows.
    """
    # TODO: implement
    pass


# TODO: implement another distance metric
