import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
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
    result = data.copy()
    for col in result.columns:
        label_encoder = LabelEncoder()
        result[col] = label_encoder.fit_transform(result[col])
    # TODO: Remove this sampling step
    result = result.iloc[:100]
    return result


def preprocess_hepatitis_datasets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hepatitis dataset without modifying the original.
    Column names and shape are unmodified in the result.

    input:
        data: pd.DataFrame - dataframe to preprocess

    output:
        pd.DataFrame - preprocessed dataframe
    """
    numerical_columns = ["BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"]
    for col in numerical_columns:
        knn_imputer = KNNImputer(n_neighbors=5)
        standard_scaler = StandardScaler()
        data[col] = standard_scaler.fit_transform(knn_imputer.fit_transform(data[[col]]))

    for col in list(set(data.columns) - set(numerical_columns) - set(["Class"])):
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])
    data["Class"] = data["Class"].apply(lambda x: 1 if x == b"DIE" else 0)
    return data
