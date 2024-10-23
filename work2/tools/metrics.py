from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

def cross_validate(
    estimator: BaseEstimator,
    train_dfs: list[pd.DataFrame],
    test_dfs: list[pd.DataFrame],
    target_col: str,
    use_proba: bool = False,
    score_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
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
    actuals_list = []
    preds_list = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        estimator.fit(X_train, y_train)
        if use_proba:
            preds_list.append(estimator.predict_proba(X_test))
        else:
            preds_list.append(estimator.predict(X_test))
        actuals_list.append(y_test)
    if score_func:
        return np.array([score_func(preds, actuals) for preds, actuals in zip(preds_list, actuals_list)])
    return np.concatenate(actuals_list), np.concatenate(preds_list)

import time

def train_and_evaluate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[list, list, float, float]:
    """
    Train and evaluate a machine learning model.

    This function fits the model on the training data, makes predictions on the test data,
    and measures the time taken for both training and testing.

    Args:
        model: The machine learning model to train and evaluate.
        X_train (array-like): The training input samples.
        y_train (array-like): The target values for training.
        X_test (array-like): The test input samples.
        y_test (array-like): The target values for testing.

    Returns:
        tuple: A tuple containing:
            - y_trues_all (list): True labels from the test set.
            - y_preds_all (list): Predicted labels for the test set.
            - total_train_time (float): Total time taken for training the model.
            - total_test_time (float): Total time taken for making predictions on the test set.
    """
    total_train_time = 0.0
    total_test_time = 0.0
    y_trues_all = []
    y_preds_all = []

    # Fit the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    total_train_time += train_time

    # Predict on the test set
    start_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_time
    total_test_time += test_time

    # Collect true labels and predictions
    y_trues_all.extend(y_test)
    y_preds_all.extend(y_pred)

    return y_trues_all, y_preds_all, total_train_time, total_test_time