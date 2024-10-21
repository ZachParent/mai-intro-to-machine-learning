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
