import argparse
import pandas as pd
import os
import logging
from pathlib import Path
from tools.config import METRICS_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def compute_analysis(df: pd.DataFrame, dataset_name: str, model_name: str, params: dict):
    pass


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

    filepaths = sorted(list(METRICS_DATA_DIR.glob("**/*.csv")))
    for filepath in filepaths:
        metrics_data_config = get_config_from_filepath(filepath)
        logger.info(f"Running analysis for config {metrics_data_config}")

        metrics_data = pd.read_csv(filepath)

        compute_analysis(metrics_data, **metrics_data_config)


if __name__ == "__main__":
    main()
