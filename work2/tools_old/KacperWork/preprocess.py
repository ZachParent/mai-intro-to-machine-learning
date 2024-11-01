import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import glob
import numpy as np
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
    for col in data.columns:
        if data[col].dtype == object:  # Only apply to object columns
            data[col] = data[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    le = LabelEncoder()
    scaler = MinMaxScaler()
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent', missing_values="?")

    categorical_cols = ["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
                         "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "HISTOLOGY", "Class"]
    numerical_cols = ["AGE", "BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"]

    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

    # Apply Label Encoding to each categorical column of the data
    for column in categorical_cols:
        data[column] = le.fit_transform(data[column])

    # Fill in missing entries in the numerical data using simple imputer
    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

    # Normalise the data in the numerical columns using Min-Max scaling.
    # (good in situations where distance-based algorithms will be used, we will be using KNN)
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

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

