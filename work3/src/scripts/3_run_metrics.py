import argparse

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from tools.config import CLUSTERED_DATA_DIR, PREPROCESSED_DATA_DIR, DATA_DIR
from tools.metrics import adjusted_rand_index, purity, davies_bouldin_index, f_measure

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)

def load_true_labels(dataset_name):
    preprocessed_data_path = Path(PREPROCESSED_DATA_DIR) / f"{dataset_name}.csv"
    if not preprocessed_data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {preprocessed_data_path}")
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    if 'class' not in preprocessed_data.columns:
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

    predicted_labels = df['cluster'].values

    matched_predicted_labels = hungarian_algorithm(true_labels, predicted_labels)

    ari = adjusted_rand_index(true_labels, matched_predicted_labels)
    pur = purity(true_labels, matched_predicted_labels)
    dbi = davies_bouldin_index(df.iloc[:, :-2].values, matched_predicted_labels)
    f1 = f_measure(true_labels, matched_predicted_labels)

    metrics = {
        "ARI": ari,
        "Purity": pur,
        "DBI": dbi,
        "F1 Score": f1
    }
    return pd.Series(metrics, index=metrics.keys())

def get_config_from_filepath(filepath: Path) -> dict:
    dataset_name = filepath.parent.parent.name
    model_name = filepath.parent.name
    params_str = filepath.stem.split(",")
    params = {param.split("=")[0]: param.split("=")[1] for param in params_str}
    return {
        "dataset": dataset_name,
        "model": model_name,
        "params": params,
    }

def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    filepaths = sorted(list(CLUSTERED_DATA_DIR.glob("**/*.csv")))
    output_data = []
    for filepath in filepaths:
        curr_output_data = {}
        clustered_data_config = get_config_from_filepath(filepath)
        logger.info(f"Computing metrics for config {clustered_data_config}")

        curr_output_data['model'] = clustered_data_config['model']
        curr_output_data['dataset'] = clustered_data_config['dataset']

        clustered_data = pd.read_csv(filepath)
        true_labels = load_true_labels(clustered_data_config["dataset"])
        metrics_data = compute_metrics(clustered_data, true_labels)
        curr_output_data.update(metrics_data)

        for param, value in clustered_data_config["params"].items():
            curr_output_data[param] = value

        output_data.append(curr_output_data)

    metrics_data_path = DATA_DIR / f"3_metrics.csv"
    pd.DataFrame(output_data).to_csv(metrics_data_path, index=False)

if __name__ == "__main__":
    main()
