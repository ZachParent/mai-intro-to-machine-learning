import glob
import pandas as pd
from scipy.io.arff import loadarff
import logging

logger = logging.getLogger(__name__)


def load_data(
    train_file_pattern: str, test_file_pattern: str
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Load train and test data from ARFF files matching the given patterns.

    This function reads ARFF files for both training and test datasets,
    converts them to pandas DataFrames, and returns them as lists.

    Parameters:
    -----------
    train_file_pattern : str
        Glob pattern to match training data files.
    test_file_pattern : str
        Glob pattern to match test data files.

    Returns:
    --------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        A tuple containing two lists:
        1. List of DataFrames for training data.
        2. List of DataFrames for test data.

    Note:
    -----
    The function uses scipy.io.arff.loadarff to read ARFF files and
    converts the data to pandas DataFrames. Files are sorted to ensure
    consistent ordering across runs.
    """
    train_dfs = []
    test_dfs = []
    for is_test, file_pattern in [
        (False, train_file_pattern),
        (True, test_file_pattern),
    ]:
        for file in sorted(glob.glob(file_pattern)):
            raw_data, meta = loadarff(file)
            logger.info(f"Loading {file}")
            df = pd.DataFrame(raw_data, columns=meta.names())
            if is_test:
                test_dfs.append(df)
            else:
                train_dfs.append(df)
    return train_dfs, test_dfs