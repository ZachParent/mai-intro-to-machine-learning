from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import time


def train_and_evaluate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_proba: bool = False,
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
        use_proba (bool): Whether to use the probability of the predicted class.

    Returns:
        tuple: A tuple containing:
            - y_true (list): True labels from the test set.
            - y_pred (list): Predicted labels for the test set.
            - train_time (float): Time taken for training the model.
            - test_time (float): Time taken for making predictions on the test set.
    """
    # Fit the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict on the test set
    start_time = time.time()
    if use_proba:
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test)
    test_time = time.time() - start_time
    return y_test, y_pred, train_time, test_time


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
        use_proba (bool): Whether to use the probability of the predicted class.
        score_func (Callable[[np.ndarray, np.ndarray], float]): The scoring function to use.

    Returns:
        tuple[list, list, float, float]: A tuple containing:
            - actuals_list (list): List of true labels.
            - preds_list (list): List of predicted labels.
            - train_time (float): Total time taken for training.
            - test_time (float): Total time taken for testing.

        if score_func is not None:
            return np.array([score_func(preds, actuals) for preds, actuals in zip(preds_list, actuals_list)])
    """
    actuals_list = []
    preds_list = []
    train_times = []
    test_times = []
    for train_df, test_df in zip(train_dfs, test_dfs):
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        y_true, y_pred, train_time, test_time = train_and_evaluate_model(
            estimator, X_train, y_train, X_test, y_test, use_proba
        )
        actuals_list.append(y_true)
        preds_list.append(y_pred)
        train_times.append(train_time)
        test_times.append(test_time)

    if score_func:
        return np.array(
            [score_func(preds, actuals) for preds, actuals in zip(preds_list, actuals_list)]
        )
    return actuals_list, preds_list, train_times, test_times
