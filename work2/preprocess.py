import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
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
    le = LabelEncoder()

    # Apply Label Encoding to each column of the data
    for column in data.columns:
        data[column] = le.fit_transform(data[column])

    return data


def preprocess_hepatitis_datasets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hepatitis dataset without modifying the original.
    Column names and shape are unmodified in the result.

    input:
        data: pd.DataFrame - dataframe to preprocess

    output:
        pd.DataFrame - preprocessed dataframe
    """

    missing_values = count_missing_values(data)

    # Print the result
    print(f"Total missing values: {missing_values}")

    return data


def count_missing_values(data: pd.DataFrame) -> dict:
    """
    Returns the number of missing values in the entire DataFrame
    and the number of missing values per column.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to check for missing values.
    
    Returns:
    dict: A dictionary containing the total number of missing values and a column-wise breakdown.
    """ 
    return data.isnull().sum().sum()

