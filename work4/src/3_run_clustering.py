import argparse
import pandas as pd
import os
from itertools import product
import logging
import time
from tools.config import CLUSTERED_DATA_DIR, NON_REDUCED_DATA_NAME
from tools.clustering import CLUSTERING_MODEL_MAP, CLUSTERING_PARAMS_MAP


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="The name of the model to run",
    choices=CLUSTERING_MODEL_MAP.keys(),
    required=True,
)
parser.add_argument(
    "--input_file_path",
    type=str,
    help="The path to the input data",
    required=True,
)
parser.add_argument(
    "--reduced",
    action="store_true",
    help="Whether the input data is reduced",
)
parser.add_argument(
    "--verbose", "-v", action="store_true", help="Whether to print verbose output"
)

logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    input_data = pd.read_csv(args.input_file_path)
    if args.reduced:
        input_dataset = os.path.basename(
            os.path.dirname(os.path.dirname(args.input_file_path))
        )
        reduction_method = os.path.basename(os.path.dirname(args.input_file_path))
        input_file_basename = os.path.splitext(os.path.basename(args.input_file_path))[
            0
        ]
    else:
        input_dataset = os.path.splitext(os.path.basename(args.input_file_path))[0]
        reduction_method = NON_REDUCED_DATA_NAME
        input_file_basename = ""

    features_data = input_data.iloc[:, :-1]

    clustered_data_dir = (
        CLUSTERED_DATA_DIR / input_dataset / reduction_method / args.model
    )
    os.makedirs(clustered_data_dir, exist_ok=True)

    params = CLUSTERING_PARAMS_MAP[args.model][input_dataset]

    runtimes = []
    model = CLUSTERING_MODEL_MAP[args.model](**params)

    logger.info(
        f"Running {input_dataset}/{reduction_method}/{args.model}, params: {', '.join(f'{k}={v}' for k, v in params.items())}..."
    )
    tik = time.time()
    clusters = model.fit_predict(features_data)
    tok = time.time()

    logger.info(f"Time taken: {tok - tik} seconds")
    runtime_data = {
        "dataset": input_dataset,
        "reduction_method": reduction_method,
        "clustering_model": args.model,
        **params,
        "runtime": tok - tik,
    }
    runtimes.append(runtime_data)

    clustered_data = pd.concat(
        [input_data.iloc[:, :-1], pd.Series(clusters, name="cluster")], axis=1
    )

    clustered_data_path = (
        clustered_data_dir
        / f"{input_file_basename + ',' if input_file_basename else ''}{','.join(f'{k}={v}' for k, v in params.items())}.csv"
    )
    clustered_data.to_csv(clustered_data_path, index=False)

    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(clustered_data_dir / "runtime.csv", index=False)


if __name__ == "__main__":
    main()
