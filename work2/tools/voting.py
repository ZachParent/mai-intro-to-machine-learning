from abc import ABC, abstractmethod
import numpy as np
from collections import Counter


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
        # Use Counter to count occurrences of each class and select the most common one
        class_counts = Counter(classes)
        most_common_class = class_counts.most_common(1)[0][0]
        return most_common_class


class InverseDistanceWeightedVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of inverse distances to the rows.
        """
        class_weights = {}
        for distance, class_label in zip(distances, classes):
            if distance == 0:
                # If the distance is zero, treat it as infinitely close (weight = infinity)
                return class_label
            weight = 1 / distance
            class_weights[class_label] = class_weights.get(class_label, 0) + weight

        # Return the class with the highest accumulated weight
        return max(class_weights, key=class_weights.get)


class ShepardsWorkVote(VotingFunc):
    def __call__(self, distances: list[np.number], classes: list[int]) -> int:
        """
        Returns the class that has the smallest sum of squared distances to the rows,
        using an exponential instead of an inverse distance scheme.
        """
        class_weights = {}
        for distance, class_label in zip(distances, classes):
            weight = np.exp(-(distance**2))  # Exponential decay based on squared distance
            class_weights[class_label] = class_weights.get(class_label, 0) + weight

        # Return the class with the highest accumulated weight
        return max(class_weights, key=class_weights.get)
