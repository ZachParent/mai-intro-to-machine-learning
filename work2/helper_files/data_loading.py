
# .py file relating to loading data from the ARFF files

import pandas as pd
from scipy.io import arff
import os

# Function to load ARFF files and return a pandas DataFrame
def load_arff_file(file_path):
    data = arff.loadarff(file_path)
    return pd.DataFrame(data)

# Function to load all ARFF files in a directory
def load_all_arff_files(directory):
    all_data = [] 
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".arff"): 
            file_path = os.path.join(directory, filename)

            df = load_arff_file(file_path)

            # Add the resulting df to the list
            all_data.append(df) 
    
    return all_data

# Function to combine all of the arff files that are part of a dataset
def combine_arff_files(dataset):
    return pd.concat(dataset, ignore_index=True)