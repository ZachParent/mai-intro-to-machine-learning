import numpy as np
from .knn import KNNClassifier


def RNN(X_train, y_train, distance_func):
    """
    Reduced Nearest Neighbor (RNN) algorithm for instance reduction.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      distance_func: Distance function to use (e.g., EuclideanDistance()).

    Returns:
      Reduced training data features and labels.
    """
    reduced_X = [X_train[0]]  # Start with the first instance
    reduced_y = [y_train[0]]

    for i in range(1, len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        knn = KNNClassifier(k=1, distance_func=distance_func) 
        knn.fit(reduced_X, reduced_y)
        pred_y_i = knn.predict([x_i])[0]

        if pred_y_i != y_i:  # If the reduced set misclassifies the instance
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

        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train) 
        neighbors = knn.kneighbors([x_i], return_distance=False)[0]

        neighbor_labels = [y_train[n] for n in neighbors]
        class_probabilities = [neighbor_labels.count(c) / k for c in np.unique(y_train)]

        if max(class_probabilities) >= threshold and y_i == neighbor_labels[0]:  # Keep if probability is above threshold and agrees with nearest neighbor
            reduced_X.append(x_i)
            reduced_y.append(y_i)

    return np.array(reduced_X), np.array(reduced_y)


def IB2(X_train, y_train, distance_func):
    """
    IB2 (Instance-Based 2) hybrid reduction algorithm.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      distance_func: Distance function to use.

    Returns:
      Reduced training data features and labels.
    """
    # 1. Start with the first instance
    reduced_X = [X_train[0]]
    reduced_y = [y_train[0]]

    # 2. Iterate through the remaining instances
    for i in range(1, len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        # 3. Classify the current instance using the reduced set
        knn = KNNClassifier(k=1, distance_func=distance_func)
        knn.fit(reduced_X, reduced_y)
        pred_y_i = knn.predict([x_i])[0]

        # 4. If misclassified, add it to the reduced set
        if pred_y_i != y_i:
            reduced_X.append(x_i)
            reduced_y.append(y_i)

    # 5. (Edited NN step) - Remove instances that are misclassified by their nearest neighbor in the reduced set
    for i in range(len(reduced_X)):
        x_i = reduced_X[i]
        y_i = reduced_y[i]

        knn = KNNClassifier(k=1, distance_func=distance_func)
        knn.fit(reduced_X, reduced_y)
        neighbors = knn.kneighbors([x_i], return_distance=False)[0]
        nearest_neighbor_label = reduced_y[neighbors[0]]

        if nearest_neighbor_label != y_i:
            reduced_X = np.delete(reduced_X, i, axis=0)
            reduced_y = np.delete(reduced_y, i)

    return np.array(reduced_X), np.array(reduced_y)


def reductionKNNAlgorithm(X_train, y_train, X_test, k, distance_func, reduction_method):
    """
    kNN algorithm with instance reduction preprocessing.

    Args:
      X_train: Training data features.
      y_train: Training data labels.
      X_test: Test data features.
      k: Number of nearest neighbors.
      distance_func: Distance function to use.
      reduction_method: The reduction method to apply ('RNN', 'ENNTh', 'IB2').

    Returns:
      Predictions for the test data.
    """

    if reduction_method == 'RNN':
        X_train_reduced, y_train_reduced = RNN(X_train, y_train, distance_func)
    elif reduction_method == 'ENNTh':
        X_train_reduced, y_train_reduced = ENNTh(X_train, y_train)
    elif reduction_method == 'IB2':
        X_train_reduced, y_train_reduced = IB2(X_train, y_train, distance_func)
    else:
        raise ValueError("Invalid reduction method.")

    knn = KNNClassifier(k=k, distance_func=distance_func)
    knn.fit(X_train_reduced, y_train_reduced)
    predictions = knn.predict(X_test)

    storage_percentage = (len(X_train_reduced) / len(X_train)) * 100

    return predictions, storage_percentage