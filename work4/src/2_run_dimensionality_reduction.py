import argparse
import pandas as pd
import os
from itertools import product
import logging
import time
from tools.pca import PCA
from tools.config import PREPROCESSED_DATA_DIR, REDUCED_DATA_DIR
from tools.clustering import MODEL_MAP, PARAMS_GRID_MAP

METHOD_MAP = {
    "pca": PCA,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The name of the dataset to run the model on",
    choices=["hepatitis", "mushroom", "vowel", "synthetic"],
    required=True,
)
parser.add_argument(
    "--method",
    type=str,
    help="The name of the dimensionality reduction method to run",
    choices=METHOD_MAP.keys(),
    required=True,
)
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    preprocessed_data_path = PREPROCESSED_DATA_DIR / f"{args.dataset}.csv"
    preprocessed_data = pd.read_csv(preprocessed_data_path).iloc[:, :50]

    features_data = preprocessed_data.iloc[:, :-1]

    reduced_data_dir = REDUCED_DATA_DIR / args.dataset
    os.makedirs(reduced_data_dir, exist_ok=True)

    runtimes = []

    logger.info(
        f"Running {args.dataset}/{args.method}..."
    )

    method = METHOD_MAP[args.method]()

    tik = time.time()
    reduced_data = method.fit_transform(features_data)
    tok = time.time()

    logger.info(f"Time taken: {tok - tik} seconds")
    runtime_data = {
        "dataset": args.dataset,
        "method": args.model,
        "runtime": tok - tik,
    }
    runtimes.append(runtime_data)

    reduced_data_path = reduced_data_dir / f"{args.dataset}_{args.method}.csv"
    reduced_data.to_csv(reduced_data_path, index=False)

    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(reduced_data_dir / f"{args.dataset}_{args.method}_runtime.csv", index=False)


if __name__ == "__main__":
    main()
