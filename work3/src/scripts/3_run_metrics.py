import argparse
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
from tools.config import CLUSTERED_DATA_DIR, METRICS_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def compute_metrics(
    df: pd.DataFrame, dataset_name: str, model_name: str, params: dict
) -> pd.DataFrame:
    return pd.DataFrame(np.ones(df.shape[1]))


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
