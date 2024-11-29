from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, f1_score

# Internal Metrics
def davies_bouldin_index(X, predicted_labels):
    unique_labels = np.unique(predicted_labels)
    if len(unique_labels) <= 1:
        return float("nan")
    return davies_bouldin_score(X, predicted_labels)

def calinski_harabasz_index(X, predicted_labels):
    unique_labels = np.unique(predicted_labels)
    if len(unique_labels) <= 1:
        return float("nan")
    return calinski_harabasz_score(X, predicted_labels)



# External Metrics
def adjusted_rand_index(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def f_measure(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average="macro")
