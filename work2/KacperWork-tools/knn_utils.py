import numpy as np
import matplotlib.pyplot as plt
from tools.knn import KNNClassifier
from tools.distance import EuclideanDistance, ManhattanDistance, MahalanobisDistance
from tools.voting import InverseDistanceWeightedVote, MajorityClassVote, ShepardsWorkVote
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn_relief import ReliefF
from sklearn.metrics import precision_recall_curve, average_precision_score

def prepare_data(train_dfs, test_dfs):
    X_train = train_dfs[0].drop(columns=['Class']).to_numpy()
    y_train = train_dfs[0]['Class'].to_numpy()
    
    X_test = test_dfs[0].drop(columns=['Class']).to_numpy()
    y_test = test_dfs[0]['Class'].to_numpy()
    
    return X_train, y_train, X_test, y_test


def calculate_feature_weights(X_train, y_train):
    # Calculate Information Gain (mutual information)
    info_gain_weights = mutual_info_classif(X_train, y_train)
    normalized_info_gain_weights = info_gain_weights / np.sum(info_gain_weights)  # Normalize to sum to 1
    
    # Calculate feature importance using ReliefF
    relief = ReliefF()
    relief.fit(X_train, y_train)
    relief_weights = relief.w_
    normalized_relief_weights = relief_weights / np.sum(relief_weights)  # Normalize to sum to 1
    
    return normalized_info_gain_weights, normalized_relief_weights


def find_best_params(X_train, y_train, k_values, distance_functions, voting_schemes, weights_list):
    best_score = 0
    best_params = {}

    for k in k_values:
        for distance_func in distance_functions:
            for voting_func in voting_schemes:
                for weights in weights_list:
                    print(f"Testing combination: k={k}, distance_func={distance_func.__class__.__name__}, voting_func={voting_func.__class__.__name__}, weights={'Equal' if weights is None else 'Custom'}")
                    
                    knn = KNNClassifier(k=k, distance_func=distance_func, voting_func=voting_func, weights=weights)
                    
                    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                    avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'k': k,
                            'distance_func': distance_func.__class__.__name__,
                            'voting_func': voting_func.__class__.__name__,
                            'weights': 'Equal' if weights is None else 'Custom'
                        }

    return best_score, best_params


def plot_precision_recall_curve(knn, X_train, y_train):
    # Use cross_val_predict to get predicted probabilities for the positive class
    y_scores = cross_val_predict(knn, X_train, y_train, cv=5, method='predict_proba')
    y_scores_positive = y_scores[:, 1]  # Probabilities for the positive class

    precision, recall, thresholds = precision_recall_curve(y_train, y_scores_positive)

    # Calculate the average precision score (AP score)
    average_precision = average_precision_score(y_train, y_scores_positive)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def train_and_evaluate_final_model(knn_final, X_train, y_train, X_test, y_test):
    knn_final.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn_final.predict(X_test)

    # Calculate accuracy on the test set
    accuracy_test = np.mean(y_pred == y_test)
    print(f"Test Set Accuracy: {accuracy_test:.4f}")
