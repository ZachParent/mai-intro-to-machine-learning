import numpy as np

from tools.knn import KNNClassifier


# Condensed: GCNN
def GCNN(X, y, k: int, knn: KNNClassifier):
    classes = np.unique(y)
    X_reduced = []
    y_reduced = []

    # Select one sample per class for the initial set
    for c in classes:
        idx = np.where(y == c)[0][0]  # Select the first occurrence of class c
        X_reduced.append(X.iloc[idx])
        y_reduced.append(y[idx])

    X_reduced = np.array(X_reduced)
    y_reduced = np.array(y_reduced)

    # Iteratively add misclassified samples to set
    while True:
        knn.fit(X_reduced, y_reduced)

        # Track if any points were added in this iteration
        additions = False

        for i in range(len(X)):
            # Skip points already in set
            # Correct comparison to check if the row already exists in set
            if any(np.all(X_reduced == X.iloc[i].to_numpy(), axis=1)):
                continue

            # Check if current point is misclassified by KNN
            pred = knn.predict([X.iloc[i]])
            if pred != y[i]:
                # Add this misclassified sample to set
                X_reduced = np.vstack([X_reduced, X.iloc[i].to_numpy()])
                y_reduced = np.append(y_reduced, y[i])
                additions = True

        # Stop if no new points were added to set
        if not additions:
            break

    return X_reduced, y_reduced


# Edited: ENNTH
def ENNTH(X, y, k: int, knn: KNNClassifier):

    threshold = 0.6  # For threshold of 0.5, no reduction occured --> Set threshold to 0.6

    knn.fit(X.values, y)

    indices_to_keep = []

    # Get predicted probabilities for each instance
    probabilities = knn.predict_proba(X.values)

    # Determine which instances to keep based on the threshold
    for i in range(len(X)):
        if np.max(probabilities[i]) > threshold:
            indices_to_keep.append(i)

    X_reduced = X.iloc[indices_to_keep]
    y_reduced = y[indices_to_keep]

    return X_reduced, y_reduced


# Hybrid: Drop3
def drop3(X, y, k: int, knn: KNNClassifier):
    # Use ENN to remove noisy instances
    X, y = edited_nearest_neighbor(X, y, k, knn)

    to_keep = set(range(len(X)))

    # Evaluate each instance for potential removal
    for i in range(len(X)):
        if i not in to_keep:
            continue

        # Fit KNN on all instances except the current one
        X_temp = np.delete(X, list(to_keep - {i}), axis=0)
        y_temp = np.delete(y, list(to_keep - {i}), axis=0)

        knn.fit(X_temp, y_temp)

        # Check classification accuracy of neighbors without the current instance
        correctly_classified_without_i = True
        for neighbor in list(to_keep - {i}):
            if knn.predict([X.iloc[neighbor]])[0] != y.iloc[neighbor]:
                correctly_classified_without_i = False
                break

        # If removing instance i does not affect classification, remove it
        if correctly_classified_without_i:
            to_keep.remove(i)

    X_reduced = X.iloc[list(to_keep)]
    y_reduced = y.iloc[list(to_keep)]

    return X_reduced, y_reduced


def edited_nearest_neighbor(X, y, k: int, knn: KNNClassifier):
    to_keep = []

    for i in range(len(X)):
        # Fit KNN on all instances except the current one
        X_temp = np.delete(X, i, axis=0)
        y_temp = np.delete(y, i, axis=0)
        knn.fit(X_temp, y_temp)

        # Predict the class of the current instance
        if knn.predict([X.iloc[i]])[0] == y.iloc[i]:
            to_keep.append(i)

    return X.iloc[to_keep], y.iloc[to_keep]
