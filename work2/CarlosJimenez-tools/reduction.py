import numpy as np
from .knn import KNNClassifier

def RNN(X_train, y_train, model):
    """
    Reduced Nearest Neighbor (RNN) algorithm for instance reduction.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      model: The classifier model to use (kNN or SVM).

    Returns:
      Reduced training data features and labels.
    """
    reduced_X = [X_train[0]]
    reduced_y = [y_train[0]]

    for i in range(1, len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        model.fit(reduced_X, reduced_y)
        pred_y_i = model.predict([x_i])[0]

        if pred_y_i != y_i:
            reduced_X.append(x_i)
            reduced_y.append(y_i)

    return np.array(reduced_X), np.array(reduced_y)


def ENNTh(X_train, y_train, k=3, threshold=0.5):
    """
    Edited Nearest Neighbor with Threshold (ENNTh) algorithm for instance reduction.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      k: Number of nearest neighbors to consider.
      threshold: Probability threshold for editing.

    Returns:
      Reduced training data features and labels.
    """
    reduced_X = []
    reduced_y = []

    for i in range(len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        knn = KNeighborsClassifier(n_neighbors=k)  # Using a temporary kNN for ENNTh
        knn.fit(X_train, y_train)
        neighbors = knn.kneighbors([x_i], return_distance=False)[0]

        neighbor_labels = [y_train[n] for n in neighbors]
        class_probabilities = [neighbor_labels.count(c) / k for c in np.unique(y_train)]

        if max(class_probabilities) >= threshold and y_i == neighbor_labels[0]:
            reduced_X.append(x_i)
            reduced_y.append(y_i)

    return np.array(reduced_X), np.array(reduced_y)


def IB2(X_train, y_train, model):
    """
    IB2 (Instance-Based 2) hybrid reduction algorithm.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      model: The classifier model to use (kNN or SVM).

    Returns:
      Reduced training data features and labels.
    """
    reduced_X = [X_train[0]]
    reduced_y = [y_train[0]]

    for i in range(1, len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        model.fit(reduced_X, reduced_y)
        pred_y_i = model.predict([x_i])[0]

        if pred_y_i != y_i:
            reduced_X.append(x_i)
            reduced_y.append(y_i)

    for i in range(len(reduced_X)):
        x_i = reduced_X[i]
        y_i = reduced_y[i]

        model.fit(reduced_X, reduced_y)  # Re-fit the model for each instance
        neighbors = model.kneighbors([x_i], n_neighbors=1, return_distance=False)[0]
        nearest_neighbor_label = reduced_y[neighbors[0]]

        if nearest_neighbor_label != y_i:
            reduced_X = np.delete(reduced_X, i, axis=0)
            reduced_y = np.delete(reduced_y, i)

    return np.array(reduced_X), np.array(reduced_y)


def reductionKNNAlgorithm(model, train_dfs, test_dfs, target_col, reduction_method):
    """
    kNN or SVM algorithm with instance reduction preprocessing and cross-validation.

    Args:
      model: The classifier model to use (kNN or SVM).
      train_dfs: List of training DataFrames.
      test_dfs: List of testing DataFrames.
      target_col: Name of the target column.
      reduction_method: The reduction method to apply ('RNN', 'ENNTh', 'IB2').

    Returns:
      A list of scores for each fold after cross-validation.
    """
    scores = []

    for i in range(len(train_dfs)):
        X_train = train_dfs[i].drop(columns=[target_col]).values
        y_train = train_dfs[i][target_col].values
        X_test = test_dfs[i].drop(columns=[target_col]).values
        y_test = test_dfs[i][target_col].values

        if reduction_method == 'RNN':
            X_train_reduced, y_train_reduced = RNN(X_train, y_train, model)
        elif reduction_method == 'ENNTh':
            X_train_reduced, y_train_reduced = ENNTh(X_train, y_train)
        elif reduction_method == 'IB2':
            X_train_reduced, y_train_reduced = IB2(X_train, y_train, model)
        else:
            raise ValueError("Invalid reduction method.")

        model.fit(X_train_reduced, y_train_reduced)
        predictions = model.predict(X_test)
        score = f1_score(y_test, predictions, average="weighted") 
        scores.append(score)

    return scores