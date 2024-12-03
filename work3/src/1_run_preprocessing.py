import argparse
import pandas as pd
import scipy.io.arff
import os
import logging
from tools.config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from tools.preprocess import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    datasets = ["hepatitis", "mushroom", "vowel"]

    for dataset in datasets:
        logger.info(f"Preprocessing {dataset}")

        raw_data_path = RAW_DATA_DIR / f"{dataset}.arff"
        raw_data, meta = scipy.io.arff.loadarff(raw_data_path)

        df = pd.DataFrame(raw_data, columns=meta.names())

        # Decode byte-strings if necessary
        df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)

        preprocessed_data = preprocess_dataset(df)

        preprocessed_data_path = PREPROCESSED_DATA_DIR / f"{dataset}.csv"
        preprocessed_data.to_csv(preprocessed_data_path, index=False)


if __name__ == "__main__":
    main()
