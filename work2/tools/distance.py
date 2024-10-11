import numpy as np


from abc import ABC, abstractmethod

class DistanceFunc(ABC):
    """A distance function takes in two rows and returns a distance."""

    @abstractmethod
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        pass


class ManhattanDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        """
        Calculates the Manhattan distance between two rows.
        """
        # Sanity check
        if row1.shape != row2.shape:
            raise ValueError("Shape of the two rows need to be the same")
        
        return np.abs(row1 - row2).sum()


class EuclideanDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        """
        Calculates the Euclidean distance between two rows.
        """
        # Sanity check
        if row1.shape != row2.shape:
            raise ValueError("Shape of the two rows need to be the same")
        
        # Works because Euclidian dist is the L2 norm
        return np.linalg.norm(row1 - row2)


# TODO: implement another distance metric as a class inheriting from DistanceFunc

# NOTES ON WHICH METRIC TO IMPLEMENT:

