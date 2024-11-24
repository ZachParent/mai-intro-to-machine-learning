import argparse
import pandas as pd
import os
import logging
from tools.config import METRICS_DATA_PATH
from tools.clustering import PARAMS_GRID_MAP

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def compute_analysis(metrics_data: pd.Series, metrics_data_config: dict):
    print(metrics_data_config)
    print(metrics_data)
    pass


def get_metrics_from_row(row: pd.Series) -> pd.Series:
    return row.loc[["ari", "purity", "dbi", "f_measure"]]

def get_config_from_row(row: pd.Series) -> dict:
    params_keys = PARAMS_GRID_MAP[row["model"]].keys()
    return {
        "dataset_name": row["dataset"],
        "model_name": row["model"],
        "params": {
            key: row[key] for key in params_keys
        },
    }


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    metrics_data = pd.read_csv(METRICS_DATA_PATH)
    for _, row in metrics_data.iterrows():
        metrics_data_config = get_config_from_row(row)
        metrics_data = get_metrics_from_row(row)
        logger.info(f"Running analysis for config {metrics_data_config}")


        compute_analysis(metrics_data, metrics_data_config)


if __name__ == "__main__":
    main()
