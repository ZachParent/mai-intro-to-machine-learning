from abc import ABC, abstractmethod
import numpy as np


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
        # TODO: implement
        pass


class InverseDistanceWeightedVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of inverse distances to the
        rows.
        """
        # TODO: implement
        pass


class ShepardsWorkVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of squared distances to the
        rows, using an exponential instead of an inverse distance scheme.
        """
        # TODO: implement
        pass
