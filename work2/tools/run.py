import os
import itertools
import time
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from tools.metrics import train_and_evaluate_model, cross_validate
from tools.knn import KNNClassifier
from tools.distance import ManhattanDistance, EuclideanDistance, ChebyshevDistance, MahalanobisDistance
from tools.voting import MajorityClassVote, InverseDistanceWeightedVote, ShepardsWorkVote
from tools.preprocess import preprocess_hepatitis_datasets, load_datasets, preprocess_mushroom_datasets
from tools.weighting import InformationGainWeighting, ReliefFWeighting, EqualWeighting
from tools.reduction import GCNN, ENNTH, drop3, edited_nearest_neighbor
import numpy as np
from sklearn.svm import SVC as SVMClassifier

import argparse
import logging

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR.parent, 'data')

def run_svm(train_dfs: List[pd.DataFrame], 
            test_dfs: List[pd.DataFrame], 
            dataset_name: str, 
            class_columns_per_ds: Dict[str, str]) -> None:
    """
    Run SVM classification with various parameter configurations.
    """
    # SVM Parameters
    c_values = [1, 3, 5, 7]
    kernel_types = ["linear", "poly", "rbf", "sigmoid"]

    # Results DF
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
            "test_time"
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "C",
            "kernel_type",
            "fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"
        ]
    )   

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
            f1_scores.append(f1_score(y_true, y_pred))

        # Compute confusion matrix and metrics
        cm = confusion_matrix(y_trues_all, y_preds_all)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_trues_all, y_preds_all)
        f1 = f1_score(y_trues_all, y_preds_all)

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
            total_test_time
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            C,
            kernel_type,
            *f1_scores
        ]

    # Save the results for SVM
    file_name = f'svm_{dataset_name}'
    cross_validated_results_file_path = os.path.join(DATA_DIR, "cross_validated_results", f'{file_name}.csv')
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f'{file_name}.csv')
    per_fold_results.to_csv(per_fold_results_file_path, index=False)

def run_knn(train_dfs: List[pd.DataFrame], 
            test_dfs: List[pd.DataFrame], 
            dataset_name: str, 
            class_columns_per_ds: Dict[str, str], 
            full_data_X: pd.DataFrame, 
            full_data_y: pd.Series) -> Tuple[KNNClassifier, Dict[str, np.ndarray]]:
    """
    Run KNN classification with various parameter configurations.
    """
    # Define CM for MahalanobisDistance
    # TODO: fix this for MahalanobisDistance
    # covariance_matrix = np.cov(train_dfs[0].drop(columns=[class_columns_per_ds[dataset_name]]).apply(pd.to_numeric, errors='coerce'),
    #                         rowvar=False)

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
            "test_time"
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"
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
            f"Running KNN: [weighting_func={weighting_func.__class__.__name__}, distance_func={distance_func.__class__.__name__}, voting_func={voting_func.__class__.__name__}, k={k}]")

        y_trues_list, y_preds_list, train_time, test_time = cross_validate(knn, train_dfs, test_dfs, class_columns_per_ds[dataset_name])
        f1_scores = [f1_score(y_trues, y_preds) for y_trues, y_preds in zip(y_trues_list, y_preds_list)]
        
        y_trues = np.concatenate(y_trues_list)
        y_preds = np.concatenate(y_preds_list)

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
            test_time
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            k,
            distance_func.__class__.__name__,
            voting_func.__class__.__name__,
            weighting_func.__class__.__name__,
            *f1_scores
        ]

        # Store best configuration instance
        if best_config_instance is None or accuracy > cross_validated_results['accuracy'].max():
            best_config_instance = {
                "k": k,
                "distance_func": distance_func,  # Save the instance
                "voting_func": voting_func,
                "weighting_func": weighting_func
            }

    # Save the results for KNN
    file_name = f'knn_{dataset_name}'
    cross_validated_results_file_path = os.path.join(DATA_DIR, "cross_validated_results", f'{file_name}.csv')
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f'{file_name}.csv')
    per_fold_results.to_csv(per_fold_results_file_path, index=False)

    return best_config_instance, weights

def run_reduced_knn(train_dfs: List[pd.DataFrame], 
                    test_dfs: List[pd.DataFrame], 
                    dataset_name: str, 
                    class_columns_per_ds: Dict[str, str], 
                    best_config_instance: Dict[str, any], 
                    weights: Dict[str, np.ndarray]) -> None:
    """
    Run reduced KNN classification with various techniques.
    """
    best_k = best_config_instance["k"]
    best_distance_func = best_config_instance["distance_func"]
    best_voting_func = best_config_instance["voting_func"]
    best_weighting_func = best_config_instance["weighting_func"]

    reduction_funcs = {"control": lambda x, y, z, s: (x, y), "GGCN": GCNN, "ENNTH":ENNTH, "Drop3": drop3}
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
            "storage"
        ]
    )
    per_fold_results = pd.DataFrame(
        columns=[
            "k",
            "distance_func",
            "voting_func",
            "weighting_func",
            "reduction_func",
            "fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"
        ]
    )

    best_k = 7
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
        for train_df, test_df in zip(train_dfs, test_dfs):
            X_train = train_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_train = train_df[class_columns_per_ds[dataset_name]]
            X_test = test_df.drop(columns=[class_columns_per_ds[dataset_name]])
            y_test = test_df[class_columns_per_ds[dataset_name]]

            logging.debug(f"Reducing training data with {reduction_func}")
            X_train_reduced, y_train_reduced = reduction_funcs[reduction_func](X_train, y_train, best_k, knn)

            storage = len(X_train_reduced)

            # Train and evaluate the KNN model
            y_true, y_pred, train_time, test_time = train_and_evaluate_model(knn, X_train_reduced, y_train_reduced,
                                                                                X_test,
                                                                                y_test)
            f1_scores.append(f1_score(y_true, y_pred))
            # Update totals
            total_train_time += train_time
            total_test_time += test_time
            total_storage += storage
            # Collect true labels and predictions
            y_trues_all.extend(y_true)
            y_preds_all.extend(y_pred)
            logging.debug(f"Reduced training data storage: {len(X_train_reduced)} / {len(X_train)}. Took {train_time} seconds to train and {test_time} seconds to test.")

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
            total_storage/10 #average storage over folds
        ]
        per_fold_results.loc[len(per_fold_results)] = [
            best_k,
            best_distance_func.__class__.__name__,
            best_voting_func.__class__.__name__,
            best_weighting_func.__class__.__name__,
            reduction_func,
            *f1_scores
        ]

    # Save the results for reduced KNN
    file_name = f'knn_reduction_{dataset_name}'
    cross_validated_results_file_path = os.path.join(DATA_DIR, "cross_validated_results", f'{file_name}.csv')
    cross_validated_results.to_csv(cross_validated_results_file_path, index=False)
    per_fold_results_file_path = os.path.join(DATA_DIR, "per_fold_results", f'{file_name}.csv')
    per_fold_results.to_csv(per_fold_results_file_path, index=False)


def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the model with a specific dataset.')
    parser.add_argument('--dataset_name', type=str, choices=['hepatitis', 'mushroom'], required=True,
                        help='The name of the dataset to use (hepatitis or mushroom)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Whether to print verbose output')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='How many samples to use from the dataset')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Configurations
    dataset_name = args.dataset_name
    processing_funcs_per_ds = {"hepatitis": preprocess_hepatitis_datasets, "mushroom": preprocess_mushroom_datasets}
    class_columns_per_ds = {"hepatitis": "Class", "mushroom": "class"}

    logging.debug(f"Using {dataset_name} dataset...")
    train_path = f'{DATA_DIR}/raw/{dataset_name}/*train.arff'
    test_path = f'{DATA_DIR}/raw/{dataset_name}/*test.arff'
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
    run_svm(train_dfs, test_dfs, dataset_name, class_columns_per_ds)

    # ========== KNN ==========

    logging.info("Running KNN...")
    best_config_instance, weights = run_knn(train_dfs, test_dfs, dataset_name, class_columns_per_ds, full_data_X, full_data_y)

    # ========== Reduced KNN ==========

    logging.info("Running reduced KNN...")
    run_reduced_knn(train_dfs, test_dfs, dataset_name, class_columns_per_ds, best_config_instance, weights)

    logging.info(f"Finished in {time.time() - start_time} seconds")

if __name__ == '__main__':
    run()
