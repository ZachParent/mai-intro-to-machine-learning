import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def identify_categorical_and_numerical(df, unique_threshold=0.05):
    """
    Identifies categorical and numerical columns in a DataFrame.

    Args:
        df: The pandas DataFrame.
        unique_threshold: The threshold for the proportion of unique values
                           to consider a column categorical.

    Returns:
        A tuple containing two lists: categorical columns and numerical columns.
    """
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if df[col].dtype == "object" or df[col].nunique() / len(df) <= unique_threshold:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    return categorical_cols, numerical_cols


def preprocess_dataset(df):
    # Replace ? with nan for correct imputation
    df.replace("?", np.nan, inplace=True)

    class_col_name = df.columns[-1]
    class_col = df[class_col_name]
    df.drop(class_col_name, axis=1, inplace=True)

    # Get categorical and numerical columns based on dtype and heuristics
    categorical_cols, numeric_cols = identify_categorical_and_numerical(df)
    logger.info(f"categorical columns: {categorical_cols}")
    logger.info(f"numeric columns: {numeric_cols}")

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="mean"),
                        ),  # Fill missing with mean
                        ("scaler", MinMaxScaler()),  # Min-Max scaling
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),  # Fill missing with mode
                        # Use 'passthrough' to handle categorical data
                        ("passthrough", "passthrough"),
                    ]
                ),
                categorical_cols,
            ),
        ],
    )

    processed_array = preprocessor.fit_transform(df)

    binary_cols = [col for col in categorical_cols if df[col].nunique() <= 2]
    non_binary_cols = [col for col in categorical_cols if df[col].nunique() > 2]

    processed_numeric_df = pd.DataFrame(
        processed_array[:, : len(numeric_cols)], columns=numeric_cols
    )
    processed_categorical_df = pd.DataFrame(
        processed_array[:, len(numeric_cols) :], columns=categorical_cols
    )

    encoded_categorical_binary_df = processed_categorical_df.drop(
        non_binary_cols, axis=1
    )

    # Label Encoder
    for col in binary_cols:
        le = LabelEncoder()
        encoded_categorical_binary_df[col] = le.fit_transform(
            encoded_categorical_binary_df[col]
        )

    # One-hot encode categorical features (the literature suggests OneHotEcoding for k-means clustering approaches )
    ohe = OneHotEncoder(sparse_output=False)
    encoded_categorical_non_binary_array = ohe.fit_transform(
        processed_categorical_df[non_binary_cols]
    )  # Encode categorical part

    # Rename encoded columns
    encoded_columns = []
    for i, cat_col in enumerate(non_binary_cols):
        for j in range(ohe.categories_[i].size):
            encoded_columns.append(f"{cat_col}_{ohe.categories_[i][j]}")

    encoded_categorical_non_binary_df = pd.DataFrame(
        encoded_categorical_non_binary_array, columns=encoded_columns
    )

    final_df = pd.concat(
        [
            processed_numeric_df,
            encoded_categorical_binary_df,
            encoded_categorical_non_binary_df,
        ],
        axis=1,
    )

    # Encoding the class column (for metrics)
    le = LabelEncoder()
    final_df["class"] = le.fit_transform(class_col)

    return final_df
