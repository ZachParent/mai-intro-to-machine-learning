import os
import itertools

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from tools.metrics import train_and_evaluate_model
from tools.knn import KNNClassifier
from tools.distance import ManhattanDistance, EuclideanDistance, ChebyshevDistance, MahalanobisDistance
from tools.voting import MajorityClassVote, InverseDistanceWeightedVote, ShepardsWorkVote
from tools.preprocess import preprocess_hepatitis_datasets, load_datasets, preprocess_mushroom_datasets
from tools.weighting import InformationGainWeighting, ReliefFWeighting, EqualWeighting
import numpy as np
from sklearn.svm import SVC as SVMClassifier

import argparse
import logging

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR.parent, 'data')


def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the model with a specific dataset.')
    parser.add_argument('--dataset_name', type=str, choices=['hepatitis', 'mushroom'],
                        help='The name of the dataset to use (hepatitis or mushroom)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Whether to print verbose output')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Configurations
    dataset_name = args.dataset_name
    processing_funcs_per_ds = {"hepatitis": preprocess_hepatitis_datasets, "mushroom": preprocess_mushroom_datasets}
    class_columns_per_ds = {"hepatitis": "Class", "mushroom": "class"}

    logging.debug(f"Running {dataset_name}...")
    train_path = f'{DATA_DIR}/raw/{dataset_name}/*train.arff'
    test_path = f'{DATA_DIR}/raw/{dataset_name}/*test.arff'
    train_dfs = load_datasets(train_path)
    test_dfs = load_datasets(test_path)

    logging.debug(f"train_path: {train_path}")
    train_dfs = [processing_funcs_per_ds[dataset_name](df) for df in train_dfs]
    test_dfs = [processing_funcs_per_ds[dataset_name](df) for df in test_dfs]
    logging.debug(f"Train datasets count: {len(train_dfs)}")
    logging.debug(f"Test datasets count: {len(test_dfs)}")

    # ========== SVM ==========

    # SVM Parameters
    c_values = [1, 3, 5, 7]
    kernel_types = ["linear", "poly", "rbf", "sigmoid"]

    # Results DF
    results_svm = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "accuracy",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time"
        ]
    )

    # Run all parameter configurations
    for C, kernel_type in itertools.product(c_values, kernel_types):
        svm = SVMClassifier(
            C=C,
            kernel=kernel_type,
        )

        logging.info(f"Running SVM: [kernel={kernel_type}, C={C}]")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time = 0.0, 0.0

        # Cross-validate
        for train_df, test_df in zip(train_dfs, test_dfs):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            # Train and evaluate the SVM model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(svm, X_train, y_train, X_test, y_test)

            # Update totals
            total_train_time += train_time
            total_test_time += test_time

            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)

        # Compute confusion matrix and metrics
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)

        # Append the results
        results_svm.loc[len(results_svm)] = [
            C,
            kernel_type,
            accuracy,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time
        ]

    # Save the results for SVM
    file_path_svm = os.path.join(DATA_DIR, "results", f'svm_{dataset_name}.csv')
    results_svm.to_csv(file_path_svm, index=False)

    # ========== KNN ==========

    # Define CM for MahalanobisDistance
    covariance_matrix = np.cov(train_dfs[0].drop(columns=["Class"]).apply(pd.to_numeric, errors='coerce'),
                                rowvar=False)

    # KNN Parameters
    k_values = [1, 3, 5, 7]
    distance_funcs = [ManhattanDistance(), EuclideanDistance(), ChebyshevDistance(),
                        MahalanobisDistance(covariance_matrix)]
    voting_funcs = [MajorityClassVote(), InverseDistanceWeightedVote(), ShepardsWorkVote()]
    weighting_funcs = [EqualWeighting(), InformationGainWeighting(), ReliefFWeighting()]

    # Results DF
    results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "accuracy",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time"
        ]
    )

    # Run all parameter configurations
    best_config_instance = None
    for k, distance_func, voting_func, weighting_func in itertools.product(
            k_values, distance_funcs, voting_funcs, weighting_funcs
    ):
        knn = KNNClassifier(
            k=k,
            distance_func=distance_func,
            voting_func=voting_func,
            weights=None,
        )

        logging.info(
            f"Running KNN: [weighting_func={weighting_func.__class__.__name__}, distance_func={distance_func.__class__.__name__}, voting_func={voting_func.__class__.__name__}, k={k}]")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time = 0.0, 0.0

        # Cross-validate
        for train_df, test_df in zip(train_dfs, test_dfs):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            # Fit weighting function
            weighting_func.fit(X_train, y_train)
            weights = weighting_func.get_weights()

            # Pass the weights to the KNN classifier
            knn.set_weights(weights)

            # Train and evaluate the KNN model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(knn, X_train, y_train, X_test, y_test)
            # Update totals
            total_train_time += train_time
            total_test_time += test_time

            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)

        # Append the results
        results.loc[len(results)] = [
            k,
            distance_func.__class__.__name__,
            voting_func.__class__.__name__,
            weighting_func.__class__.__name__,
            accuracy,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time
        ]

        # Store best configuration instance
        if best_config_instance is None or accuracy > results['accuracy'].max():
            best_config_instance = {
                "k": k,
                "distance_func": distance_func,  # Save the instance
                "voting_func": voting_func,
                "weighting_func": weighting_func
            }

    # Save the results for KNN
    file_path = os.path.join(DATA_DIR, "results", f'knn_{dataset_name}.csv')
    results.to_csv(file_path, index=False)

    # Extract the best configuration instance
    best_k = best_config_instance["k"]
    best_distance_func = best_config_instance["distance_func"]
    best_voting_func = best_config_instance["voting_func"]
    best_weighting_func = best_config_instance["weighting_func"]

    print(
        f"Best configuration: k={best_k}, distance_func={best_distance_func.__class__.__name__}, voting_func={best_voting_func.__class__.__name__}, weighting_func={best_weighting_func.__class__.__name__}")

    print(type(best_distance_func))

    # ========== Reduced KNN ==========

    reduction_funcs = ["regular", "cnn", "enn", "drop2"]
    reduction_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "reduction_func",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time",
            "storage"
        ]
    )

    for reduction_func in reduction_funcs:
        knn = KNNClassifier(
            k=best_k,
            distance_func=best_distance_func,  # Use instance, not string
            voting_func=best_voting_func,
            weights=None,  # We will apply weights later
        )

        print(f"Running KNN with reduction: {reduction_func}")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time = 0.0, 0.0

        train_dfs = load_datasets(train_path)
        test_dfs = load_datasets(test_path)

        train_dfs = [processing_funcs_per_ds[dataset_name](df) for df in train_dfs]
        test_dfs = [processing_funcs_per_ds[dataset_name](df) for df in test_dfs]

        for train_df, test_df in zip(train_dfs, test_dfs):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            # Apply the appropriate reduction technique
            if reduction_func == "cnn":
                X_train_reduced, y_train_reduced = knn.condensed_nearest_neighbor(X_train, y_train)
            elif reduction_func == "enn":
                X_train_reduced, y_train_reduced = knn.edited_nearest_neighbor(X_train, y_train)
            elif reduction_func == "drop2":
                X_train_reduced, y_train_reduced = knn.drop2(X_train, y_train)
            else:
                X_train_reduced, y_train_reduced = X_train, y_train

            storage = len(X_train_reduced)

            # Fit weighting function
            weighting_func.fit(X_train_reduced, y_train_reduced)
            weights = weighting_func.get_weights()

            # Pass the weights to the KNN classifier
            knn.set_weights(weights)

            # Train and evaluate the KNN model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(knn, X_train_reduced, y_train_reduced,
                                                                                X_test,
                                                                                y_test)
            # Update totals
            total_train_time += train_time
            total_test_time += test_time

            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)

        # Append the results
        reduction_results.loc[len(reduction_results)] = [
            best_k,
            best_distance_func.__class__.__name__,
            best_voting_func.__class__.__name__,
            best_weighting_func.__class__.__name__,
            reduction_func,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time,
            storage
        ]

        # Save the results for reduced KNN
        file_path_reduction = os.path.join(DATA_DIR, "results", f'knn_reduction_{dataset_name}.csv')
        reduction_results.to_csv(file_path_reduction, index=False)

if __name__ == '__main__':
    run()

