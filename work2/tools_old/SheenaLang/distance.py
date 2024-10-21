import numpy as np
from abc import ABC, abstractmethod

class DistanceFunc(ABC):
    """A distance function takes in two rows and returns a distance."""

    @abstractmethod
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        pass

class ManhattanDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        return np.sum(np.abs(row1 - row2))


class EuclideanDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        return np.sqrt(np.sum((row1 - row2) ** 2))

class ChebyshevDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        return np.max(np.abs(row1 - row2))