from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class VotingFunc(ABC):
    """
    A voting function that takes in a list of distances and a list of classes and returns a class.

    input:
        distances: list[np.number] - list of distances
        classes: list[int] - list of classes

    output:
        int - class
    """

    @abstractmethod
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        pass


class MajorityClassVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that appears most frequently in the list of rows.
        """
        return np.argmax(np.bincount(classes))


class InverseDistanceWeightedVote(VotingFunc):

    def __init__(self, distance_weight: float = 1.0):
        self.distance_weight = distance_weight

    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of inverse distances to the
        rows.
        """
        inverse_distances = [1 / (d ** self.distance_weight) for d in distances]
        class_sums = {}
        for cls, inv_dist in zip(classes, inverse_distances):
            class_sums[cls] = class_sums.get(cls, 0) + inv_dist
        return min(class_sums, key=class_sums.get)


class ShepardsWorkVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of squared distances to the
        rows, using an exponential instead of an inverse distance scheme.
        """
        # TODO: implement
        pass
