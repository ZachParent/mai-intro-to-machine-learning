import argparse
import pandas as pd
import os
from itertools import product
import logging
import time
from tools.dimensionality_reduction import PCA, KernelPCA, PCA_PARAMS_GRID_MAP, KERNEL_PCA_PARAMS_GRID_MAP
from tools.config import PREPROCESSED_DATA_DIR, REDUCED_DATA_DIR

METHOD_MAP = {
    "pca": PCA,
    "kernel_pca": KernelPCA,
}

PARAMS_GRID_MAP = {
    "pca": PCA_PARAMS_GRID_MAP,
    "kernel_pca": KERNEL_PCA_PARAMS_GRID_MAP,
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
    preprocessed_data = pd.read_csv(preprocessed_data_path)

    features_data = preprocessed_data.iloc[:, :-1]

    reduced_data_dir = REDUCED_DATA_DIR / args.dataset / args.method
    os.makedirs(reduced_data_dir, exist_ok=True)

    runtimes = []
    params_grid = PARAMS_GRID_MAP[args.method]
    for params in product(*params_grid.values()):
        param_dict = dict(zip(params_grid.keys(), params))
        model = METHOD_MAP[args.method](**param_dict)

        logger.info(
            f"Running {args.dataset}/{args.method}, params: {', '.join(f'{k}={v}' for k, v in param_dict.items())}..."
        )
        tik = time.time()
        reduced_data = model.fit_transform(features_data)
        tok = time.time()

        logger.info(f"Time taken: {tok - tik} seconds")
        runtime_data = {
            "dataset": args.dataset,
            "method": args.method,
            **param_dict,
            "runtime": tok - tik,
        }
        runtimes.append(runtime_data)

        reduced_data = pd.DataFrame(
            reduced_data, columns=[f"dim_{i}" for i in range(reduced_data.shape[1])]
        )

        reduced_data_path = (
            reduced_data_dir / f"{','.join(f'{k}={v}' for k, v in param_dict.items())}.csv"
        )
        reduced_data.to_csv(reduced_data_path, index=False)

    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(reduced_data_dir / "runtime.csv", index=False)


if __name__ == "__main__":
    main()
