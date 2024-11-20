from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import f1_score


def adjusted_rand_index(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def purity(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)

    # For each cluster, take the count of the most common class
    total_correct = np.sum(np.amax(cm, axis=1))
    return total_correct / len(true_labels)


def davies_bouldin_index(X, predicted_labels):
    unique_labels = np.unique(predicted_labels)
    if len(unique_labels) <= 1:
        return float('nan')
    return davies_bouldin_score(X, predicted_labels)

def f_measure(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='macro')


