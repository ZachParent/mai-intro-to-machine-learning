import argparse
import pandas as pd
import os
import logging
from tools.config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", '-v', action='store_true', help="Whether to print verbose output")

logger = logging.getLogger(__name__)

def preprocess_hepatitis(df: pd.DataFrame) -> pd.DataFrame:
    return df

def preprocess_mushroom(df: pd.DataFrame) -> pd.DataFrame:
    return df

# TODO: preprocess third dataset

def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    for dataset, preprocess_func in [("hepatitis", preprocess_hepatitis), ("mushroom", preprocess_mushroom)]:
        # TODO: fix this to merge the folds
        raw_data_path = RAW_DATA_DIR / f"{dataset}.csv"
        raw_data = pd.read_csv(raw_data_path)

        preprocessed_data = preprocess_func(raw_data)

        preprocessed_data_path = PREPROCESSED_DATA_DIR / f"{dataset}.csv"
        preprocessed_data.to_csv(preprocessed_data_path, index=False)



if __name__ == "__main__":
    main()
