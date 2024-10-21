import numpy as np


from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine

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
        return np.sum(np.abs(row1 - row2))


class EuclideanDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        """
        Calculates the Euclidean distance between two rows.
        """
        return np.sqrt(np.sum((row1 - row2) ** 2))

    
class CosineDistance(DistanceFunc):
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        """
        Calculates the cosine distance between two rows.
        """
#         return cosine(row1, row2)  
        dot_product = np.dot(row1, row2)
        norm_row1 = np.linalg.norm(row1)
        norm_row2 = np.linalg.norm(row2)
        return dot_product / (norm_row1 * norm_row2) 
