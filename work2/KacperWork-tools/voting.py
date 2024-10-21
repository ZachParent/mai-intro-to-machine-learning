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
        class_array = np.array(classes)
        class_counts = np.bincount(class_array)

        return np.argmax(class_counts)


class InverseDistanceWeightedVote(VotingFunc):

    def __init__(self, distance_weight: float = 1.0):
        self.distance_weight = distance_weight

    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of inverse distances to the
        row.
        """
        distances = np.array(distances)
        classes = np.array(classes)

        inverse_distances = 1 / distances # Calculating inverse distances

        unique_classes = np.unique(classes)

        inverse_sums = np.array([np.sum(inverse_distances[classes == unique_class]) for unique_class in unique_classes])

        return unique_classes[np.argmin(inverse_sums)]


class ShepardsWorkVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of negative exponential
        distances to the row.
        """
        distances = np.array(distances)
        classes = np.array(classes)

        # Compute exponential weighting based on squared distances
        weights = np.exp(-distances**2)

        unique_classes = np.unique(classes)

        weighted_sums = np.array([np.sum(weights[classes == unique_class]) for unique_class in unique_classes])

        return unique_classes[np.argmax(weighted_sums)]
