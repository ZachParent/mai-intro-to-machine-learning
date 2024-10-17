from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

def cross_validate(
    estimator: BaseEstimator,
    train_dfs: list[pd.DataFrame],
    test_dfs: list[pd.DataFrame],
    target_col: str,
    score_func: Callable[[np.ndarray[np.number], np.ndarray[np.number]], any],
) -> np.ndarray:
    """
    Perform cross-validation for the estimator.

    Args:
        estimator (BaseEstimator): The estimator to use.
        train_dfs (list[pd.DataFrame]): List of training datasets.
        test_dfs (list[pd.DataFrame]): List of testing datasets.
        target_col (str): The name of the target column.
        score_func (Callable[[np.ndarray, np.ndarray], float]): The scoring function to use.

    Returns:
        np.ndarray: An array of scores from cross-validation.
    """
    scores = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        scores.append(score_func(y_test, preds))
    return np.array(scores)
