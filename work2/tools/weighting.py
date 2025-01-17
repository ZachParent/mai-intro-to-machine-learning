import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_relief import ReliefF  # Assuming you're using ReliefF from sklearn-relief package


class WeightingMethod(BaseEstimator, TransformerMixin):
    """
    Abstract base class for feature weighting methods.
    All subclasses should implement the `fit` and `get_weights` methods.
    """

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        raise NotImplementedError("Subclasses should implement this method!")

    def get_weights(self):
        raise NotImplementedError("Subclasses should implement this method!")


class InformationGainWeighting(WeightingMethod):
    """
    Feature weighting using Information Gain (Mutual Information).
    """

    def __init__(self):
        super().__init__()
        self.weights_ = None

    def fit(self, X_train, y_train):
        # Calculate Information Gain (mutual information)
        info_gain_weights = mutual_info_classif(X_train, y_train)
        # Normalize weights to sum to 1
        self.weights_ = info_gain_weights

    def get_weights(self):
        return self.weights_


class EqualWeighting(WeightingMethod):
    """
    Feature weighting using equal weights for all features.
    """

    def __init__(self):
        super().__init__()
        self.weights_ = None

    def fit(self, X_train, y_train):
        # Assign equal weights to all features
        n_features = X_train.shape[1]
        equal_weights = np.ones(n_features)
        # Normalize weights to sum to 1
        self.weights_ = equal_weights

    def get_weights(self):
        return self.weights_


class ReliefFWeighting(WeightingMethod):
    """
    Feature weighting using the ReliefF algorithm.
    """

    def __init__(self):
        super().__init__()
        self.weights_ = None
        self.relief = ReliefF()  # Initialize ReliefF from sklearn-relief

    def fit(self, X_train, y_train):
        # Convert X_train and y_train to numpy arrays if they are pandas DataFrames
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Fit the ReliefF model to the data
        self.relief.fit(X_train, y_train)

        # Get ReliefF feature weights
        relief_weights = self.relief.w_

        # Normalize weights to sum to 1
        if np.sum(relief_weights) == 0:
            raise ValueError("Sum of ReliefF weights is zero, cannot normalize!")
        self.weights_ = relief_weights

    def get_weights(self):
        return self.weights_
