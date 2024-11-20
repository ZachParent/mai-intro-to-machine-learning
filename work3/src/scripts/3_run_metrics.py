import argparse
import pandas as pd
import os
import logging
from pathlib import Path
from tools.config import CLUSTERED_DATA_DIR, METRICS_DATA_DIR, PREPROCESSED_DATA_DIR
from tools.metrics import adjusted_rand_index, purity, davies_bouldin_index, f_measure

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)

def load_true_labels(dataset_name):
    preprocessed_data_path = Path(PREPROCESSED_DATA_DIR) / dataset_name / f"{dataset_name}.csv"
    if not preprocessed_data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {preprocessed_data_path}")
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    if 'class' not in preprocessed_data.columns:
        raise KeyError(f"'class' column not found in the dataset: {preprocessed_data_path}")
    return preprocessed_data["class"].values

def compute_metrics(df: pd.DataFrame, dataset_name: str, model_name: str, params: dict) -> pd.DataFrame:
    true_labels = load_true_labels(dataset_name)

    predicted_labels = df['cluster'].values

    ari = adjusted_rand_index(true_labels, predicted_labels)
    pur = purity(true_labels, predicted_labels)
    dbi = davies_bouldin_index(df.iloc[:, :-2].values, predicted_labels)  # Assuming df includes features for DBI
    f1 = f_measure(true_labels, predicted_labels)

    metrics = {
        "ARI": ari,
        "Purity": pur,
        "DBI": dbi,
        "F1 Score": f1
    }
    return pd.DataFrame(metrics, index=[0])

def get_config_from_filepath(filepath: Path) -> dict:
    dataset_name = filepath.parent.parent.name
    model_name = filepath.parent.name
    params_str = filepath.stem.split(",")
    params = {param.split("=")[0]: param.split("=")[1] for param in params_str}
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "params": params,
    }

def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    os.makedirs(METRICS_DATA_DIR, exist_ok=True)

    filepaths = sorted(list(CLUSTERED_DATA_DIR.glob("**/*.csv")))
    for filepath in filepaths:
        clustered_data_config = get_config_from_filepath(filepath)
        logger.info(f"Computing metrics for config {clustered_data_config}")

        clustered_data = pd.read_csv(filepath)

        metrics_data = compute_metrics(clustered_data, **clustered_data_config)

        metrics_data_dir = (
            METRICS_DATA_DIR
            / clustered_data_config["dataset_name"]
            / clustered_data_config["model_name"]
        )
        os.makedirs(metrics_data_dir, exist_ok=True)
        metrics_data_path = metrics_data_dir / f"{filepath.stem}.csv"
        metrics_data.to_csv(metrics_data_path, index=False)

if __name__ == "__main__":
    main()
