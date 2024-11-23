import argparse
import pandas as pd
import os
from itertools import product
import logging
from tools.clustering import (
    KMeans,
    FuzzyCMeans,
    GMeans,
    GlobalKMeans,
    Optics,
    SpectralClustering,
    KMeansParamsGrid,
    FuzzyCMeansParamsGrid,
    GMeansParamsGrid,
    GlobalKmeansParams,
    OpticsParamsGrid,
    SpectralClusteringParamsGrid
)
from tools.config import PREPROCESSED_DATA_DIR, CLUSTERED_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The name of the dataset to run the model on",
    choices=["hepatitis", "mushroom","vowel","synthetic"],
    required=True,
)
parser.add_argument(
    "--model",
    type=str,
    help="The name of the model to run",
    choices=[
        "kmeans",
        "fuzzy_cmeans",
        "gmeans",
        "global_kmeans",
        "optics",
        "spectral_clustering",
    ],
    required=True,
)
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)

model_map = {
    "kmeans": KMeans,
    "fuzzy_cmeans": FuzzyCMeans,
    "gmeans": GMeans,
    "global_kmeans": GlobalKMeans,
    "optics": Optics,
    "spectral_clustering": SpectralClustering,
}
params_grid_map = {
    "kmeans": KMeansParamsGrid,
    "fuzzy_cmeans": FuzzyCMeansParamsGrid,
    "gmeans": GMeansParamsGrid,
    "global_kmeans": GlobalKmeansParams,
    "optics": OpticsParamsGrid,
    "spectral_clustering": SpectralClusteringParamsGrid,
}


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    preprocessed_data_path = PREPROCESSED_DATA_DIR / f"{args.dataset}.csv"
    preprocessed_data = pd.read_csv(preprocessed_data_path).iloc[:, :50]

    features_data = preprocessed_data.iloc[:, :-1]

    clustered_data_dir = CLUSTERED_DATA_DIR / args.dataset / args.model
    os.makedirs(clustered_data_dir, exist_ok=True)

    params_grid = params_grid_map[args.model]
    for params in product(*params_grid.values()):
        param_dict = dict(zip(params_grid.keys(), params))
        model = model_map[args.model](**param_dict)

        logger.info(
            f"Running model {args.model} with params: {', '.join(f'{k}={v}' for k, v in param_dict.items())}"
        )

        clusters = model.fit_predict(features_data)

        clustered_data = pd.concat(
            [preprocessed_data.iloc[:, :-1], pd.Series(clusters, name="cluster")], axis=1
        )

        clustered_data_path = clustered_data_dir / f"{','.join(f'{k}={v}' for k, v in param_dict.items())}.csv"
        clustered_data.to_csv(clustered_data_path, index=False)


if __name__ == "__main__":
    main()
