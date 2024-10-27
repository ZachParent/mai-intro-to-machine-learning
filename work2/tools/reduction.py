from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from tools import KNNClassifier, MajorityClassVote, InformationGainWeighting, ManhattanDistance



def GCNN(X: pd.Dataframe, y: pd.Series, k: int, knn: KNNClassifier):
    # Step 1: Initialize Prototype Set (PS) with one sample from each class
    classes = np.unique(y)
    PS_X = []
    PS_y = []

    # Select one sample per class for the initial PS
    for c in classes:
        idx = np.where(y == c)[0][0]  # Select the first occurrence of class c
        PS_X.append(X[idx])
        PS_y.append(y[idx])

    # Convert lists to numpy arrays
    PS_X = np.array(PS_X)
    PS_y = np.array(PS_y)

    # Step 2: Iteratively add misclassified samples to PS
    while True:
        knn.fit(PS_X, PS_y)

        # Track if any points were added in this iteration
        additions = False

        for i in range(len(X)):
            # Skip points already in PS
            if any((PS_X == X[i]).all(1)):
                continue

            # Check if current point is misclassified by k-NN using PS
            pred = knn.predict([X[i]])
            if pred != y[i]:
                # Add this misclassified sample to PS
                PS_X = np.vstack([PS_X, X[i]])
                PS_y = np.append(PS_y, y[i])
                additions = True

        # Stop if no new points were added to PS
        if not additions:
            break

    return PS_X, PS_y

#Edited: RNGE

def find_relative_neighbors(X, k):
    """Find the relative neighborhood for each instance in X."""
    n = len(X)
    relative_neighbors = {i: [] for i in range(n)}

    for i in range(n):
        # Calculate distances to all other points
        distances = [(j, euclidean_distance(X[i], X[j])) for j in range(n) if j != i]
        distances.sort(key=lambda x: x[1])  # Sort by distance
        nearest_neighbors = [j for j, _ in distances[:k]]  # Select k-nearest neighbors

        # Check relative neighborhood condition for each nearest neighbor
        for j in nearest_neighbors:
            is_relative_neighbor = True
            for m, dist_m in distances:
                if m == j:
                    continue
                if euclidean_distance(X[i], X[m]) < max(euclidean_distance(X[i], X[j]), euclidean_distance(X[j], X[m])):
                    is_relative_neighbor = False
                    break
            if is_relative_neighbor:
                relative_neighbors[i].append(j)

    return relative_neighbors

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def RNGE(X, y, k):
    """Perform Relative Neighborhood Graph Editing (RNGE) on the dataset X with labels y."""
    relative_neighbors = find_relative_neighbors(X, k)

    # Collect indices to keep in the final edited set
    to_keep = []

    for i in range(len(X)):
        # Get the labels of relative neighbors
        neighbor_labels = [y[j] for j in relative_neighbors[i]]

        # If the majority label of neighbors is the same as the instance label, keep it
        if neighbor_labels:
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
            if y[i] == most_common_label:
                to_keep.append(i)
        else:
            # If there are no neighbors, keep the point as it's isolated
            to_keep.append(i)

    # Create the reduced dataset
    X_reduced = X[to_keep]
    y_reduced = y[to_keep]

    return X_reduced, y_reduced


#Hybrid: DROP3
def edited_nearest_neighbor(X, y, k, knn):
    """Perform Edited Nearest Neighbor (ENN) filtering on the dataset."""
    to_keep = []

    for i in range(len(X)):
        # Fit KNN on all instances except the current one
        X_temp = np.delete(X, i, axis=0)
        y_temp = np.delete(y, i, axis=0)
        knn.fit(X_temp, y_temp)

        # Predict the class of the current instance
        if knn.predict([X[i]])[0] == y[i]:
            to_keep.append(i)

    # Return the filtered dataset
    return X[to_keep], y[to_keep]


def drop3(X, y, k, knn):
    """Perform DROP3 reduction on the dataset."""
    # Step 1: Use ENN to remove noisy instances
    X, y = edited_nearest_neighbor(X, y, k, knn)

    # Step 2: Initialize the set of instances to be kept
    to_keep = set(range(len(X)))

    # Step 3: Evaluate each instance for potential removal
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
            if knn.predict([X[neighbor]])[0] != y[neighbor]:
                correctly_classified_without_i = False
                break

        # If removing instance i does not affect classification, remove it
        if correctly_classified_without_i:
            to_keep.remove(i)

    # Step 4: Return the reduced dataset
    X_reduced = X[list(to_keep)]
    y_reduced = y[list(to_keep)]

    return X_reduced, y_reduced

#####################################################################

def condensed_nearest_neighbor(X: pd.DataFrame, y: pd.Series, k: int):
    X_reduced = pd.DataFrame(columns=X.columns)  # Initialize an empty DataFrame with the same columns as X
    y_reduced = pd.Series(dtype=y.dtype)  # Initialize an empty Series for y_reduced

    # Start with the first instance
    X_reduced = pd.concat([X_reduced, X.iloc[[0]]], ignore_index=True)
    y_reduced = pd.concat([y_reduced, pd.Series([y.iloc[0]])], ignore_index=True)

    for i in range(1, len(X)):
        distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), X_reduced.values)
        nearest_labels = y_reduced.iloc[np.argsort(distances[0])[:k]]

        # If the instance is misclassified, add it to the reduced set
        if nearest_labels.value_counts().get(y.iloc[i], 0) <= k / 2:
            X_reduced = pd.concat([X_reduced, X.iloc[[i]]], ignore_index=True)
            y_reduced = pd.concat([y_reduced, pd.Series([y.iloc[i]])], ignore_index=True)

    return X_reduced.reset_index(drop=True), y_reduced.reset_index(drop=True)

def edited_nearest_neighbor(X: pd.DataFrame, y: pd.Series, k: int):
    # Edited Nearest Neighbor reduction
    X_reduced_list = []
    y_reduced_list = []

    for i in range(len(X)):
        distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), X.values)
        k_indices = np.argsort(distances[0])[:k]
        k_nearest_labels = y.iloc[k_indices]

        if k_nearest_labels.value_counts().get(y.iloc[i], 0) > k / 2:
            X_reduced_list.append(X.iloc[i])
            y_reduced_list.append(y.iloc[i])

    # Concatenate at once at the end
    X_reduced = pd.DataFrame(X_reduced_list).reset_index(drop=True)
    y_reduced = pd.Series(y_reduced_list).reset_index(drop=True)

    return X_reduced, y_reduced

def drop2(X: pd.DataFrame, y: pd.Series, k: int):
    # Drop2 reduction
    X_reduced_list = []
    y_reduced_list = []

    for i in range(len(X)):
        distances = pairwise_distances(X.iloc[i].values.reshape(1, -1), X.values)
        k_indices = np.argsort(distances[0])[:k]
        k_nearest_labels = y.iloc[k_indices]

        # Count of the nearest neighbors with the same label
        count_same_label = (k_nearest_labels == y.iloc[i]).sum()
        count_diff_label = k - count_same_label

        # Condition for removal: if more than half of the neighbors are different
        if count_diff_label <= count_same_label:
            X_reduced_list.append(X.iloc[i])
            y_reduced_list.append(y.iloc[i])

    # Concatenate at once at the end
    X_reduced = pd.DataFrame(X_reduced_list).reset_index(drop=True)
    y_reduced = pd.Series(y_reduced_list).reset_index(drop=True)

    return X_reduced, y_reduced