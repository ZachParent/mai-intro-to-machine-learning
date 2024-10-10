import pandas as pd
from scipy.io import arff
import glob
import logging

logger = logging.getLogger(__name__)


def load_datasets(file_pattern: str) -> list[pd.DataFrame]:
    """
    Loads data from arff files matching the given pattern.

    input:
        file_pattern: str - pattern to match arff files

    output:
        list[pd.DataFrame] - list of dataframes
    """
    files = glob.glob(file_pattern)
    dfs = []
    for file in files:
        raw_data, meta = arff.loadarff(file)
        logger.info(f"Loading {file}")
        df = pd.DataFrame(raw_data, columns=meta.names())
        dfs.append(df)
    return dfs


def preprocess_mushrooms_datasets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the mushrooms dataset without modifying the original.
    Column names and shape are unmodified in the result.

    input:
        data: pd.DataFrame - dataframe to preprocess

    output:
        pd.DataFrame - preprocessed dataframe
    """
    pass


def preprocess_hepatitis_datasets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hepatitis dataset without modifying the original.
    Column names and shape are unmodified in the result.

    input:
        data: pd.DataFrame - dataframe to preprocess

    output:
        pd.DataFrame - preprocessed dataframe
    """
    pass
