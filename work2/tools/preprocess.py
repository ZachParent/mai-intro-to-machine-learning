import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import glob
import pandas as pd
from scipy.io import arff
import logging

# Set up logger
logger = logging.getLogger(__name__)


def load_datasets(file_pattern: str) -> list[pd.DataFrame]:
    files = glob.glob(file_pattern)

    # List to store the dataframes
    dfs = []

    # Loop through each matching file
    for file in files:
        # Load the ARFF file
        raw_data, meta = arff.loadarff(file)

        # Log the file being loaded
        logger.debug(f"Loading {file}")

        # Convert the ARFF data to a pandas DataFrame
        df = pd.DataFrame(raw_data, columns=meta.names())

        # Decode byte-strings if necessary
        df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)

        # Append dataframe to the list
        dfs.append(df)

    return dfs


def preprocess_hepatitis_datasets(df):
    # Replace ? with nan for correct imputation
    df.replace('?', np.nan, inplace=True)

    # Define the numeric and categorical columns
    numeric_cols = ['AGE', 'ALK_PHOSPHATE', 'SGOT', 'BILIRUBIN', 'ALBUMIN', 'PROTIME']
    categorical_cols = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE',
                        'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE',
                        'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY', 'Class']

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing with mean
                ('scaler', MinMaxScaler())  # Min-Max scaling
            ]), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
                # Use 'passthrough' to handle categorical data
                ('passthrough', 'passthrough')
            ]), categorical_cols)
        ],
    )

    # Apply the preprocessor to the dataframe, excluding ignored columns
    processed_array = preprocessor.fit_transform(df)

    # Convert processed array back to a DataFrame
    processed_df = pd.DataFrame(processed_array, columns=numeric_cols + categorical_cols)

    # Label encoding for categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])

    # Include ignored columns in the final DataFrame
    final_df = pd.concat([processed_df.reset_index(drop=True)], axis=1)

    # Ensure the final DataFrame has the same column order as the original DataFrame
    final_df = final_df[df.columns]

    return final_df

def preprocess_mushroom_datasets(df):
    # Replace ? with nan for correct imputation
    df.replace('?', np.nan, inplace=True)

    # Define categorical columns
    categorical_cols = [
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises?',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat',
        'class'
    ]

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
                # Use 'passthrough' to handle categorical data
                ('passthrough', 'passthrough')
            ]), categorical_cols)
        ],
    )

    # Apply the preprocessor to the dataframe, excluding ignored columns
    processed_array = preprocessor.fit_transform(df)

    # Convert processed array back to a DataFrame
    processed_df = pd.DataFrame(processed_array, columns=categorical_cols)

    # Label encoding for categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])

    # Include ignored columns in the final DataFrame
    final_df = pd.concat([processed_df.reset_index(drop=True)], axis=1)

    # Ensure the final DataFrame has the same column order as the original DataFrame
    final_df = final_df[df.columns]

    return final_df