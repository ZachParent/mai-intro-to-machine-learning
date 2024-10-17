from typing import Callable
import numpy as np
import pandas as pd
from sklearn.svm import SVC as SVMClassifier


def run_svm(
    svm: SVMClassifier, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SVM algorithm on the given train and test datasets.

    Args:
        svm (SVMClassifier): The SVM model to use.
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.
        target_col (str): The name of the target column.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing predictions and actual values.
    """
    svm.fit(train_df.drop(target_col, axis=1), train_df[target_col])
    preds = svm.predict(test_df.drop(target_col, axis=1))
    actuals = test_df[target_col]
    return (preds, actuals)


def cross_validate_svm(
    svm: SVMClassifier,
    train_dfs: list[pd.DataFrame],
    test_dfs: list[pd.DataFrame],
    target_col: str,
    score_func: Callable[[np.ndarray, np.ndarray], any],
) -> np.ndarray:
    """
    Perform cross-validation for the SVM model.

    Args:
        svm (SVMClassifier): The SVM model to use.
        train_dfs (list[pd.DataFrame]): List of training datasets.
        test_dfs (list[pd.DataFrame]): List of testing datasets.
        target_col (str): The name of the target column.
        score_func (Callable[[np.ndarray, np.ndarray], float]): The scoring function to use.

    Returns:
        np.ndarray: An array of scores from cross-validation.
    """
    scores = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        preds, actuals = run_svm(svm, train_df, test_df, target_col)
        scores.append(score_func(actuals, preds))
    return np.array(scores)
