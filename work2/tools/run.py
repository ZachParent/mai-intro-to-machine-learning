import os
import itertools
import time
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from tools.metrics import train_and_evaluate_model, cross_validate
from tools.knn import KNNClassifier
from tools.distance import (
    ManhattanDistance,
    EuclideanDistance,
    ChebyshevDistance,
    MahalanobisDistance,
)
from tools.voting import MajorityClassVote, InverseDistanceWeightedVote, ShepardsWorkVote
from tools.preprocess import (
    preprocess_hepatitis_datasets,
    load_datasets,
    preprocess_mushroom_datasets,
)
from tools.weighting import InformationGainWeighting, ReliefFWeighting, EqualWeighting
from tools.reduction import GCNN, ENNTH, drop3, edited_nearest_neighbor
import numpy as np
from sklearn.svm import SVC as SVMClassifier

import argparse
import logging

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR.parent, "data")


def run_svm(
    train_dfs: List[pd.DataFrame],
    test_dfs: List[pd.DataFrame],
    dataset_name: str,
    class_columns_per_ds: Dict[str, str],
) -> None:
    """
    Run SVM classification with various parameter configurations.
    """
    # SVM Parameters
    c_values = [1, 3, 5, 7]
    kernel_types = ["linear", "poly", "rbf", "sigmoid"]

    # Results D
    cross_validated_results = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "accuracy",
            "f1",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time",
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "f1_0",
            "f1_1",
            "f1_2",
            "f1_3",
            "f1_4",
            "f1_5",
            "f1_6",
            "f1_7",
            "f1_8",
            "f1_9",
            "train_time_0",
            "train_time_1",
            "train_time_2",
            "train_time_3",
            "train_time_4",
            "train_time_5",
            "train_time_6",
            "train_time_7",
            "train_time_8",
            "train_time_9",
            "test_time_0",
            "test_time_1",
            "test_time_2",
            "test_time_3",
            "test_time_4",
            "test_time_5",
            "test_time_6",
            "test_time_7",
            "test_time_8",
            "test_time_9",
        ]
    )
    best_config_instance = None
    # Run all parameter configurations
    for C, kernel_type in itertools.product(c_values, kernel_types):
        svm = SVMClassifier(
            C=C,
            kernel=kernel_type,
        )

        logging.debug(f"Running SVM: [kernel={kernel_type}, C={C}]")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time = 0.0, 0.0
        f1_scores = []
        train_times = []
        test_times = []

        # Cross-validate    as
        for train_df, test_df in zip(train_dfs, test_dfs):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            # Train and evaluate the SVM model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(
                svm, X_train, y_train, X_test, y_test
            )

            # Update totals
            total_train_time += train_time
            total_test_time += test_time
            train_times.append(train_time)
            test_times.append(test_time)

            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)
            f1_scores.append(f1_score(y_true, y_pred))

        # Compute confusion matrix and metrics
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)
        f1 = f1_score(y_trues_all, y_preds_all)

        if best_config_instance is None or f1 > cross_validated_results["f1"].max():
            best_config_instance = {
                "C": C,
                "kernel_type": kernel_type,  # Save the instance
            }

        # Append the results
        cross_validated_results.loc[len(cross_validated_results)] = [
            C,
            kernel_type,
            accuracy,
            f1,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time,
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            C,
            kernel_type,
            *f1_scores,
            *train_times,
            *test_times,
        ]



    # Save the results for SVM
    file_name = f"svm_{dataset_name}"
    cross_validated_results_file_path = os.path.join(
        DATA_DIR, "cross_validated_results", f"{file_name}.csv"
    )
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f"{file_name}.csv")
    per_fold_results.to_csv(per_fold_results_file_path, index=False)

    return best_config_instance


def run_reduced_svm(
    train_dfs: Dict[str, List[pd.DataFrame]],
    test_dfs: List[pd.DataFrame],
    dataset_name: str,
    class_columns_per_ds: Dict[str, str],
    best_config_instance: Dict[str, any],
    reduced_data_X: Dict[str, List[pd.DataFrame]],
    reduced_data_y: Dict[str, List[pd.DataFrame]],
) -> None:

    best_C = best_config_instance["C"]
    best_kernel_type = best_config_instance["kernel_type"]

    reduction_funcs = {
        "control": lambda x, y, z, s: (x, y),
        "GGCN": GCNN,
        "ENNTH": ENNTH,
        "Drop3": drop3,
    }
    cross_validated_results = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "reduction_func",
            "accuracy",
            "f1",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time",
            "storage",
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "reduction_func",
            "f1_0",
            "f1_1",
            "f1_2",
            "f1_3",
            "f1_4",
            "f1_5",
            "f1_6",
            "f1_7",
            "f1_8",
            "f1_9",
            "train_time_0",
            "train_time_1",
            "train_time_2",
            "train_time_3",
            "train_time_4",
            "train_time_5",
            "train_time_6",
            "train_time_7",
            "train_time_8",
            "train_time_9",
            "test_time_0",
            "test_time_1",
            "test_time_2",
            "test_time_3",
            "test_time_4",
            "test_time_5",
            "test_time_6",
            "test_time_7",
            "test_time_8",
            "test_time_9",
            "storage_0",
            "storage_1",
            "storage_2",
            "storage_3",
            "storage_4",
            "storage_5",
            "storage_6",
            "storage_7",
            "storage_8",
            "storage_9",
        ]
    )

    for reduction_func in reduction_funcs:
        svm = SVMClassifier(
            C=best_C,
            kernel=best_kernel_type,
        )

        logging.debug(f"Running SVM with reduction: {reduction_func}")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time, total_storage = 0.0, 0.0, 0
        f1_scores = []
        train_times = []
        test_times = []
        storages = []
        for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            X_train_reduced = reduced_data_X[reduction_func][i]
            y_train_reduced = reduced_data_y[reduction_func][i].values.ravel()

            storage = len(X_train_reduced)

            # Train and evaluate the SVM model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(
                svm, X_train_reduced, y_train_reduced, X_test, y_test
            )

            f1_scores.append(f1_score(y_true, y_pred))
            train_times.append(train_time)
            test_times.append(test_time)
            storages.append(storage)
            # Update totals
            total_train_time += train_time
            total_test_time += test_time
            total_storage += storage
            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)
            logging.debug(
                f"Reduced training data storage: {len(X_train_reduced)} / {len(X_train)}. Took {np.sum(train_times)} seconds to train and {np.sum(test_times)} seconds to test."
            )

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)
        f1 = f1_score(y_trues_all, y_preds_all)

        # Append the results
        cross_validated_results.loc[len(cross_validated_results)] = [
            best_C,
            best_kernel_type,
            reduction_func,
            accuracy,
            f1,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time,
            total_storage / 10,  # average storage over folds
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            best_C,
            best_kernel_type,
            reduction_func,
            *f1_scores,
            *train_times,
            *test_times,
            *storages,
        ]

    # Save the results for reduced SVM
    file_name = f"svm_reduction_{dataset_name}"
    cross_validated_results_file_path = os.path.join(
        DATA_DIR, "cross_validated_results", f"{file_name}.csv"
    )
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f"{file_name}.csv")
    per_fold_results.to_csv(per_fold_results_file_path, index=False)


def run_knn(
    train_dfs: List[pd.DataFrame],
    test_dfs: List[pd.DataFrame],
    dataset_name: str,
    class_columns_per_ds: Dict[str, str],
    full_data_X: pd.DataFrame,
    full_data_y: pd.Series,
) -> Tuple[KNNClassifier, Dict[str, np.ndarray]]:
    """
    Run KNN classification with various parameter configurations.
    """

    # KNN Parameters
    k_values = [1, 3, 5, 7]
    distance_funcs = [ManhattanDistance(), EuclideanDistance(), ChebyshevDistance()]
    voting_funcs = [MajorityClassVote(), InverseDistanceWeightedVote(), ShepardsWorkVote()]
    weighting_funcs = [EqualWeighting(), InformationGainWeighting(), ReliefFWeighting()]

    cross_validated_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "accuracy",
            "f1",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time",
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "f1_0",
            "f1_1",
            "f1_2",
            "f1_3",
            "f1_4",
            "f1_5",
            "f1_6",
            "f1_7",
            "f1_8",
            "f1_9",
            "train_time_0",
            "train_time_1",
            "train_time_2",
            "train_time_3",
            "train_time_4",
            "train_time_5",
            "train_time_6",
            "train_time_7",
            "train_time_8",
            "train_time_9",
            "test_time_0",
            "test_time_1",
            "test_time_2",
            "test_time_3",
            "test_time_4",
            "test_time_5",
            "test_time_6",
            "test_time_7",
            "test_time_8",
            "test_time_9",
        ]
    )

    logging.info("Fitting weighting functions")
    weights = {}
    for weighting_func in weighting_funcs:
        weighting_func.fit(full_data_X, full_data_y)
        weights[weighting_func.__class__.__name__] = weighting_func.get_weights()

    # Run all parameter configurations
    best_config_instance = None
    for k, distance_func, voting_func, weighting_func in itertools.product(
        k_values, distance_funcs, voting_funcs, weighting_funcs
    ):

        knn = KNNClassifier(
            k=k,
            distance_func=distance_func,
            voting_func=voting_func,
            weights=weights[weighting_func.__class__.__name__],
        )

        logging.debug(
            f"Running KNN: [weighting_func={weighting_func.__class__.__name__}, distance_func={distance_func.__class__.__name__}, voting_func={voting_func.__class__.__name__}, k={k}]"
        )

        y_trues_list, y_preds_list, train_times, test_times = cross_validate(
            knn, train_dfs, test_dfs, class_columns_per_ds[dataset_name]
        )
        f1_scores = [
            f1_score(y_trues, y_preds) for y_trues, y_preds in zip(y_trues_list, y_preds_list)
        ]

        y_trues = np.concatenate(y_trues_list)
        y_preds = np.concatenate(y_preds_list)
        train_time = np.sum(train_times)
        test_time = np.sum(test_times)

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y_trues, y_preds)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        # Append the results
        cross_validated_results.loc[len(cross_validated_results)] = [
            k,
            distance_func.__class__.__name__,
            voting_func.__class__.__name__,
            weighting_func.__class__.__name__,
            accuracy,
            f1,
            tp,
            tn,
            fp,
            fn,
            train_time,
            test_time,
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            k,
            distance_func.__class__.__name__,
            voting_func.__class__.__name__,
            weighting_func.__class__.__name__,
            *f1_scores,
            *train_times,
            *test_times,
        ]

        # Store best configuration instance
        if best_config_instance is None or f1 > cross_validated_results["f1"].max():
            best_config_instance = {
                "k": k,
                "distance_func": distance_func,  # Save the instance
                "voting_func": voting_func,
                "weighting_func": weighting_func,
            }

    # Save the results for KNN
    file_name = f"knn_{dataset_name}"
    cross_validated_results_file_path = os.path.join(
        DATA_DIR, "cross_validated_results", f"{file_name}.csv"
    )
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f"{file_name}.csv")
    per_fold_results.to_csv(per_fold_results_file_path, index=False)

    return best_config_instance, weights


def generate_reduced_knn_datasets(
    train_dfs: List[pd.DataFrame],
    dataset_name: str,
    class_columns_per_ds: Dict[str, str],
    best_config_instance: Dict[str, any],
    weights: Dict[str, np.ndarray],
) -> List[pd.DataFrame]:

    knn = KNNClassifier(
        k=best_config_instance["k"],
        distance_func=best_config_instance["distance_func"],
        voting_func=best_config_instance["voting_func"],
        weights=weights[best_config_instance["weighting_func"].__class__.__name__],
    )

    reduction_funcs = {
        "control": lambda x, y, z, s: (x, y),
        "GGCN": GCNN,
        "ENNTH": ENNTH,
        "Drop3": drop3,
    }

    X_reduced_datasets = {method: [] for method in reduction_funcs.keys()}
    y_reduced_datasets = {method: [] for method in reduction_funcs.keys()}

    for i, train_df in enumerate(train_dfs):
        for method, reduction_func in reduction_funcs.items():

            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]

            X_train_reduced, y_train_reduced = reduction_func(
                X_train, y_train, best_config_instance["k"], knn
            )

            X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=X_train.columns)
            y_train_reduced_df = pd.DataFrame(
                y_train_reduced, columns=[class_columns_per_ds[dataset_name]]
            )

            X_reduced_datasets[method].append(X_train_reduced_df)
            y_reduced_datasets[method].append(y_train_reduced_df)

            output_filename_X = f"data/reduced_X/{dataset_name}_reduced_by_{method}_fold_{i}.csv"
            output_filename_y = f"data/reduced_y/{dataset_name}_reduced_by_{method}_fold_{i}.csv"

            os.makedirs(os.path.dirname(output_filename_X), exist_ok=True)
            os.makedirs(os.path.dirname(output_filename_y), exist_ok=True)

            X_train_reduced_df.to_csv(output_filename_X, index=False)
            y_train_reduced_df.to_csv(output_filename_y, index=False, header=True)


def load_reduced_datasets(
    dataset_name: str, reduction_methods: List[str]
) -> Dict[str, List[pd.DataFrame]]:
    num_folds = 10
    reduced_data_X = {method: [] for method in reduction_methods}
    reduced_data_y = {method: [] for method in reduction_methods}

    for method in reduction_methods:
        for fold in range(num_folds):
            filename_X = f"data/reduced_X/{dataset_name}_reduced_by_{method}_fold_{fold}.csv"
            filename_y = f"data/reduced_y/{dataset_name}_reduced_by_{method}_fold_{fold}.csv"

            if os.path.exists(filename_X) and os.path.exists(filename_y):
                X_fold = pd.read_csv(filename_X)
                y_fold = pd.read_csv(filename_y)

                reduced_data_X[method].append(X_fold)
                reduced_data_y[method].append(y_fold)
            else:
                print(
                    f"Warning: Missing file(s) for method '{method}' and fold {fold}. Skipping this fold."
                )

    return reduced_data_X, reduced_data_y


def run_reduced_knn(
    train_dfs: List[pd.DataFrame],
    test_dfs: List[pd.DataFrame],
    dataset_name: str,
    class_columns_per_ds: Dict[str, str],
    best_config_instance: Dict[str, any],
    weights: Dict[str, np.ndarray],
    reduced_data_X: Dict[str, List[pd.DataFrame]],
    reduced_data_y: Dict[str, List[pd.DataFrame]],
) -> None:
    """
    Run reduced KNN classification with various techniques.
    """
    best_k = best_config_instance["k"]
    best_distance_func = best_config_instance["distance_func"]
    best_voting_func = best_config_instance["voting_func"]
    best_weighting_func = best_config_instance["weighting_func"]

    reduction_funcs = {
        "control": lambda x, y, z, s: (x, y),
        "GGCN": GCNN,
        "ENNTH": ENNTH,
        "Drop3": drop3,
    }
    cross_validated_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "reduction_func",
            "accuracy",
            "f1",
            "TP",
            "TN",
            "FP",
            "FN",
            "train_time",
            "test_time",
            "storage",
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "reduction_func",
            "f1_0",
            "f1_1",
            "f1_2",
            "f1_3",
            "f1_4",
            "f1_5",
            "f1_6",
            "f1_7",
            "f1_8",
            "f1_9",
            "train_time_0",
            "train_time_1",
            "train_time_2",
            "train_time_3",
            "train_time_4",
            "train_time_5",
            "train_time_6",
            "train_time_7",
            "train_time_8",
            "train_time_9",
            "test_time_0",
            "test_time_1",
            "test_time_2",
            "test_time_3",
            "test_time_4",
            "test_time_5",
            "test_time_6",
            "test_time_7",
            "test_time_8",
            "test_time_9",
            "storage_0",
            "storage_1",
            "storage_2",
            "storage_3",
            "storage_4",
            "storage_5",
            "storage_6",
            "storage_7",
            "storage_8",
            "storage_9",
        ]
    )

    for reduction_func in reduction_funcs:
        knn = KNNClassifier(
            k=best_k,
            distance_func=best_distance_func,  # Use instance, not string
            voting_func=best_voting_func,
            weights=weights[best_weighting_func.__class__.__name__],  # We will apply weights later
        )

        logging.debug(f"Running KNN with reduction: {reduction_func}")

        y_trues_all, y_preds_all = [], []
        total_train_time, total_test_time, total_storage = 0.0, 0.0, 0
        f1_scores = []
        train_times = []
        test_times = []
        storages = []
        for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            y_train_reduced = pd.Series(reduced_data_y[reduction_func][i].values.ravel())
            X_train_reduced = reduced_data_X[reduction_func][i]

            storage = len(X_train_reduced)

            # Train and evaluate the KNN model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(
                knn, X_train_reduced, y_train_reduced, X_test, y_test
            )
            f1_scores.append(f1_score(y_true, y_pred))
            train_times.append(train_time)
            test_times.append(test_time)
            storages.append(storage)
            # Update totals
            total_train_time += train_time
            total_test_time += test_time
            total_storage += storage
            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)
            logging.debug(
                f"Reduced training data storage: {len(X_train_reduced)} / {len(X_train)}. Took {np.sum(train_times)} seconds to train and {np.sum(test_times)} seconds to test."
            )

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)
        f1 = f1_score(y_trues_all, y_preds_all)

        # Append the results
        cross_validated_results.loc[len(cross_validated_results)] = [
            best_k,
            best_distance_func.__class__.__name__,
            best_voting_func.__class__.__name__,
            best_weighting_func.__class__.__name__,
            reduction_func,
            accuracy,
            f1,
            tp,
            tn,
            fp,
            fn,
            total_train_time,
            total_test_time,
            total_storage / 10,  # average storage over folds
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            best_k,
            best_distance_func.__class__.__name__,
            best_voting_func.__class__.__name__,
            best_weighting_func.__class__.__name__,
            reduction_func,
            *f1_scores,
            *train_times,
            *test_times,
            *storages,
        ]

    # Save the results for reduced KNN
    file_name = f"knn_reduction_{dataset_name}"
    cross_validated_results_file_path = os.path.join(
        DATA_DIR, "cross_validated_results", f"{file_name}.csv"
    )
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f"{file_name}.csv")
    per_fold_results.to_csv(per_fold_results_file_path, index=False)


def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the model with a specific dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hepatitis", "mushroom"],
        required=True,
        help="The name of the dataset to use (hepatitis or mushroom)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Whether to print verbose output"
    )
    parser.add_argument(
        "--sample", "-s", type=int, default=None, help="How many samples to use from the dataset"
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Configurations
    dataset_name = args.dataset_name
    processing_funcs_per_ds = {
        "hepatitis": preprocess_hepatitis_datasets,
        "mushroom": preprocess_mushroom_datasets,
    }
    class_columns_per_ds = {"hepatitis": "Class", "mushroom": "class"}

    logging.debug(f"Using {dataset_name} dataset...")
    train_path = f"{DATA_DIR}/raw/{dataset_name}/*train.arff"
    test_path = f"{DATA_DIR}/raw/{dataset_name}/*test.arff"
    train_dfs = load_datasets(train_path)
    test_dfs = load_datasets(test_path)

    if args.sample:
        train_dfs = [df.head(args.sample) for df in train_dfs]
        test_dfs = [df.head(args.sample) for df in test_dfs]

    logging.debug(f"train_path: {train_path}")
    train_dfs = [processing_funcs_per_ds[dataset_name](df) for df in train_dfs]
    test_dfs = [processing_funcs_per_ds[dataset_name](df) for df in test_dfs]
    logging.debug(f"Train datasets count: {len(train_dfs)}")
    logging.debug(f"Test datasets count: {len(test_dfs)}")

    full_data = pd.concat([train_dfs[0], test_dfs[0]])
    full_data_X = full_data.drop(columns=[class_columns_per_ds[dataset_name]])
    full_data_y = full_data[class_columns_per_ds[dataset_name]]

    start_time = time.time()

    # ========== SVM ==========

    logging.info("Running SVM...")
    best_config_instance_svm = run_svm(train_dfs, test_dfs, dataset_name, class_columns_per_ds)

    # ========== KNN ==========

    logging.info("Running KNN...")
    best_config_instance_knn, weights = run_knn(
        train_dfs, test_dfs, dataset_name, class_columns_per_ds, full_data_X, full_data_y
    )

    # === Generate Reduced Datasets ===

    logging.info("Reducing...")
    generate_reduced_knn_datasets(
        train_dfs, dataset_name, class_columns_per_ds, best_config_instance_knn, weights
    )

    logging.info("Loading reduced data...")
    reduction_methods = ["control", "GGCN", "ENNTH", "Drop3"]
    reduced_data_X, reduced_data_y = load_reduced_datasets(dataset_name, reduction_methods)

    # ========== Reduced KNN ==========

    logging.info("Running reduced KNN...")

    run_reduced_knn(
        train_dfs,
        test_dfs,
        dataset_name,
        class_columns_per_ds,
        best_config_instance_knn,
        weights,
        reduced_data_X,
        reduced_data_y,
    )

    # ========== Reduced SVM ==========

    logging.info("Running reduced SVM...")

    run_reduced_svm(
        train_dfs,
        test_dfs,
        dataset_name,
        class_columns_per_ds,
        best_config_instance_svm,
        reduced_data_X,
        reduced_data_y,
    )

    logging.info(f"Finished in {time.time() - start_time} seconds")


if __name__ == "__main__":
    run()
