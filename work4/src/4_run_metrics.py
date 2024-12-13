import argparse

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from tools.config import (
    PREPROCESSED_DATA_DIR,
    REDUCED_DATA_DIR,
    CLUSTERED_DATA_DIR,
    METRICS_DATA_PATH,
)
from tools.metrics import (
    davies_bouldin_index,
    calinski_harabasz_index,
    adjusted_rand_index,
    f_measure,
)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def load_true_labels(dataset_name):
    preprocessed_data_path = Path(PREPROCESSED_DATA_DIR) / f"{dataset_name}.csv"
    if not preprocessed_data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {preprocessed_data_path}")
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    if "class" not in preprocessed_data.columns:
        raise KeyError(f"'class' column not found in the dataset: {preprocessed_data_path}")
    return preprocessed_data["class"].values


def hungarian_algorithm(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)

    # This will return the optimal matching of cluster labels to true labels
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximizing the matching
    matched_labels = np.copy(predicted_labels)

    # Reassign predicted labels based on the optimal matching
    for i, j in zip(row_ind, col_ind):
        matched_labels[predicted_labels == j] = i

    return matched_labels


def compute_metrics(df: pd.DataFrame, true_labels: np.ndarray) -> pd.Series:
    predicted_labels = df["cluster"].values
    n_clusters = len(np.unique(predicted_labels))

    matched_predicted_labels = hungarian_algorithm(true_labels, predicted_labels)

    dbi = davies_bouldin_index(df.iloc[:, :-2].values, matched_predicted_labels)
    chi = calinski_harabasz_index(df.iloc[:, :-2].values, matched_predicted_labels)
    ari = adjusted_rand_index(true_labels, matched_predicted_labels)
    f1 = f_measure(true_labels, matched_predicted_labels)

    metrics = {"dbi": dbi, "chi": chi, "ari": ari, "f_measure": f1, "n_clusters": n_clusters}
    return pd.Series(metrics, index=metrics.keys())


def get_config_from_filepath(filepath: Path) -> dict:
    dataset_name = filepath.parent.parent.parent.name
    reduction_method = filepath.parent.parent.name
    clustering_model = filepath.parent.name
    params_str = filepath.stem.split(",")
    params = {param.split("=")[0]: param.split("=")[1] for param in params_str}
    return {
        "dataset": dataset_name,
        "reduction_method": reduction_method,
        "clustering_model": clustering_model,
        "params": params,
    }


def load_reduction_runtime_df():
    runtime_filepaths = sorted(list(REDUCED_DATA_DIR.glob("**/runtime.csv")))
    return pd.concat([pd.read_csv(filepath) for filepath in runtime_filepaths])


def load_clustering_runtime_df():
    runtime_filepaths = sorted(list(CLUSTERED_DATA_DIR.glob("**/runtime.csv")))
    return pd.concat([pd.read_csv(filepath) for filepath in runtime_filepaths])


def get_reduction_runtime(runtime_df: pd.DataFrame, clustered_data_config: dict):
    try:
        # First filter by dataset and model
        condition = (runtime_df["dataset"] == clustered_data_config["dataset"]) & (
            runtime_df["reduction_method"] == clustered_data_config["reduction_method"]
        )

        for k, v in clustered_data_config["params"].items():
            if k in runtime_df.columns:
                try:
                    value = float(v)
                    condition = condition & (runtime_df[k] == value)
                except ValueError:
                    condition = condition & (runtime_df[k].astype(str) == str(v))
            else:
                logging.warning(f"Parameter {k} not found in reduction runtime DataFrame columns")
        return runtime_df[condition]["runtime"].values[0]
    except (IndexError, KeyError) as e:
        logger.warning(f"Reduction runtime not found for config {clustered_data_config}. Error: {e}")
        return np.nan

def get_clustering_runtime(runtime_df: pd.DataFrame, clustered_data_config: dict):
    try:
        # First filter by dataset and model
        condition = (runtime_df["dataset"] == clustered_data_config["dataset"]) & (
            runtime_df["reduction_method"] == clustered_data_config["reduction_method"]
        ) & (runtime_df["clustering_model"] == clustered_data_config["clustering_model"])

        for k, v in clustered_data_config["params"].items():
            if k in runtime_df.columns:
                try:
                    value = float(v)
                    condition = condition & (runtime_df[k] == value)
                except ValueError:
                    condition = condition & (runtime_df[k].astype(str) == str(v))
            else:
                logging.warning(f"Parameter {k} not found in clustering runtime DataFrame columns")
        return runtime_df[condition]["runtime"].values[0]
    except (IndexError, KeyError) as e:
        logger.warning(f"Clustering runtime not found for config {clustered_data_config}. Error: {e}")
        return np.nan


def main():

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    reduction_runtime_df = load_reduction_runtime_df()
    clustering_runtime_df = load_clustering_runtime_df()
    filepaths = sorted(list(CLUSTERED_DATA_DIR.glob("**/*.csv")))
    filepaths = [filepath for filepath in filepaths if "runtime.csv" not in filepath.name]

    output_data = []
    for filepath in filepaths:
        clustered_data_config = get_config_from_filepath(filepath)
        logger.info(f"Computing metrics for config {clustered_data_config}")

        reduction_runtime = get_reduction_runtime(reduction_runtime_df, clustered_data_config)
        clustering_runtime = get_clustering_runtime(clustering_runtime_df, clustered_data_config)
        curr_output_data = {
            "dataset": clustered_data_config["dataset"],
            "reduction_method": clustered_data_config["reduction_method"],
            "clustering_model": clustered_data_config["clustering_model"],
            "reduction_runtime": reduction_runtime,
            "clustering_runtime": clustering_runtime,
        }

        clustered_data = pd.read_csv(filepath)
        true_labels = load_true_labels(clustered_data_config["dataset"])
        curr_metrics_data = compute_metrics(clustered_data, true_labels)

        curr_output_data.update(curr_metrics_data)

        for param, value in clustered_data_config["params"].items():
            curr_output_data[param] = value

        output_data.append(curr_output_data)

    metrics_data_path = METRICS_DATA_PATH
    pd.DataFrame(output_data).to_csv(metrics_data_path, index=False)


if __name__ == "__main__":
    main()
