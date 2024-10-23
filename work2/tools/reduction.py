import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


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