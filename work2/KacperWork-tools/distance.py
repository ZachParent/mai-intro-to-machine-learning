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


# Good when there are correlated features in dataset
class MahalanobisDistance(DistanceFunc):
    def __init__(self, covariance_matrix: np.ndarray[np.number]):
        """
        Initializes the Mahalanobis distance with the given covariance matrix.
        """
        # Inverse of the covariance matrix is required for Mahalanobis distance
        self.cov_inv = np.linalg.inv(covariance_matrix)
    
    def __call__(self, row1: np.ndarray[np.number], row2: np.ndarray[np.number]) -> np.number:
        """
        Calculates the Mahalanobis distance between two rows.
        """
        # Sanity check
        if row1.shape != row2.shape:
            raise ValueError("Shape of the two rows need to be the same")

        delta = row1 - row2
        return np.sqrt(delta.T @ self.cov_inv @ delta)
